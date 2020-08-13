#include <stddef.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_SAFE_CALL(ans) { _CUDA_SAFE_CALL((ans), __FILE__, __LINE__); }
#define CUDA_CHK_ERR() { _CUDA_CHK_ERR(__FILE__, __LINE__); }
#define MIN(x, y) (x) < (y) ? (x) : (y)

inline static void _CUDA_SAFE_CALL(cudaError_t code, const char *file, int line)
{
        if (code != cudaSuccess) {
                printf("CUDA runtime error: %s at line %d", 
                        cudaGetErrorString(code), line);
                exit(-1);
        }
}

inline static void _CUDA_CHK_ERR(const char *file, const int line)
{
	cudaError err = cudaGetLastError();
	_CUDA_SAFE_CALL(err, file, line);

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	_CUDA_SAFE_CALL(err, file, line);
};

__device__ 
static void get_interp_idx_weight(const double *x, 
        size_t sz_x, 
        const double *y, 
        size_t sz_y, 
        double xp, 
        double yp, 
        int idx_x[2],
        int idx_y[2], 
        double weight_x[2],
        double weight_y[2])
{
        double weight_sum;

        double spacing[2] = {x[1] - x[0], y[1] - y[0]};
        double ind_grid[2] = {(xp-x[0]) / spacing[0], (yp-y[0]) / spacing[1]};
        int max_ind_x = sz_x - 1;
        int max_ind_y = sz_y - 1;
        idx_x[0] = MIN((int)ind_grid[0],     max_ind_x);
        idx_x[1] = MIN((int)ind_grid[0] + 1, max_ind_x);
        idx_y[0] = MIN((int)ind_grid[1],     max_ind_y);
        idx_y[1] = MIN((int)ind_grid[1] + 1, max_ind_y);
        double x_grid[2] = {x[idx_x[0]], x[idx_x[1]]};
        double y_grid[2] = {y[idx_y[0]], y[idx_y[1]]};

        weight_x[0] = 1. - fabs(xp-x_grid[0]) / spacing[0];
        weight_x[1] = 1. - fabs(xp-x_grid[1]) / spacing[0];
        weight_sum = weight_x[0] + weight_x[1];
        weight_x[0] /= weight_sum;
        weight_x[1] /= weight_sum;

        weight_y[0] = 1. - fabs(yp-y_grid[0]) / spacing[1];
        weight_y[1] = 1. - fabs(yp-y_grid[1]) / spacing[1];
        weight_sum = weight_y[0] + weight_y[1];
        weight_y[0] /= weight_sum;
        weight_y[1] /= weight_sum;
}

__global__
void interp_lin_2d_kernel(const double *x, 
        size_t sz_x, // no. elements in x
        const double *y, 
        size_t sz_y, // no. elements in y
        const double *z, 
        double xp, 
        const double *yp, 
        size_t sz_yp, 
        double *out)
{
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        int idx_x[2], idx_y[2];
        double weight_x[2], weight_y[2];

        if (i >= sz_yp)
                return;

        get_interp_idx_weight(x, sz_x, y, sz_y, xp, yp[i], idx_x, idx_y,
                weight_x, weight_y);
        out[i] = 0.;
        out[i] += z[idx_x[0]*sz_y+idx_y[0]]*weight_x[0]*weight_y[0];
        out[i] += z[idx_x[0]*sz_y+idx_y[1]]*weight_x[0]*weight_y[1];
        out[i] += z[idx_x[1]*sz_y+idx_y[0]]*weight_x[1]*weight_y[0];
        out[i] += z[idx_x[1]*sz_y+idx_y[1]]*weight_x[1]*weight_y[1];
}

extern "C" {
void interp_lin_2d(const double *x, 
        size_t sz_x, // no. elements in x
        const double *y, 
        size_t sz_y, // no. elements in y
        const double *z, 
        double xp, 
        const double *yp, 
        size_t sz_yp, 
        double *out)
{
        static double *x_gpu = NULL, *y_gpu = NULL, *z_gpu = NULL;
        static double *yp_gpu, *out_gpu;
        size_t free, total;
        size_t alloc_unit = 1024*1024*1024;     // 1 GB gpu memory
        alloc_unit *= 4;
        size_t yp_pos = 0;
        size_t yp_unit = alloc_unit / sizeof(double);   // batch process unit

        if (!x_gpu) {
                CUDA_SAFE_CALL(cudaMalloc(&x_gpu, sz_x * sizeof(double)));
                CUDA_SAFE_CALL(cudaMalloc(&y_gpu, sz_y * sizeof(double)));
                CUDA_SAFE_CALL(cudaMalloc(&z_gpu, sz_x*sz_y * sizeof(double)));

                CUDA_SAFE_CALL(cudaMemcpy(x_gpu, x, sz_x*sizeof(double), cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(y_gpu, y, sz_y*sizeof(double), cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(z_gpu, z, sz_x*sz_y*sizeof(double), cudaMemcpyHostToDevice));
        }
        if (!yp_gpu) {
                CUDA_SAFE_CALL(cudaMemGetInfo(&free, &total));
                if (free < alloc_unit * 2) {
                        printf("WARNING: available cuda memory is %f MB\n", 
                                (float)free / 1024 / 1024);
                        alloc_unit = free / 2;
                }
                CUDA_SAFE_CALL(cudaMalloc(&yp_gpu, alloc_unit));
                CUDA_SAFE_CALL(cudaMalloc(&out_gpu, alloc_unit));
        }

        while (yp_pos < sz_yp) {
                size_t batch = sz_yp - yp_pos > yp_unit ? yp_unit : sz_yp - yp_pos;
                CUDA_SAFE_CALL(cudaMemcpy(yp_gpu, yp+yp_pos, batch*sizeof(double), cudaMemcpyHostToDevice));

                dim3 gridSize((sz_yp + 1024 - 1) / 1024);
                dim3 blockSize(1024);
                interp_lin_2d_kernel<<<gridSize, blockSize>>>(x_gpu, sz_x, y_gpu, sz_y,
                        z_gpu, xp, yp_gpu, batch, out_gpu);
        
                CUDA_SAFE_CALL(cudaMemcpy(out+yp_pos, out_gpu, batch*sizeof(double), cudaMemcpyDeviceToHost));
                yp_pos += batch;
        }
        CUDA_CHK_ERR();
}
}
