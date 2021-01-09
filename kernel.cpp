#include <math.h>

#define MAX(a, b) (a) > (b) ? (a) : (b)
#define NORM(x, y, z) sqrt((x)*(x)+(y)*(y)+(z)*(z))
#define DOT(ax,ay,az,bx,by,bz) ((ax)*(bx)+(ay)*(by)+(az)*(bz))

extern "C" {
void build_kernel(
        float *data,
        const int32_t *indices,
        const int64_t *indptr,
        size_t rows,
        const double *train_x,
        const double *x,
        double l_pts,
        double l_dir,
        double power_pts,
        double power_dir)
{
        const size_t stride = 6;

#pragma omp parallel for num_threads(39)
        for (size_t row = 0; row < rows; row++) {
                int64_t cols = indptr[row];
                int64_t cole = indptr[row+1];
                for (size_t i = cols; i < cole; i++) {
                        int32_t col = indices[i];
                        double view_dist;
                        double kernel_pos, kernel_view;

                        // covariance of position
                        double pos_dist = NORM(
                                train_x[stride*col]   - x[stride*row],
                                train_x[stride*col+1] - x[stride*row+1],
                                train_x[stride*col+2] - x[stride*row+2]
                        );
                        kernel_pos = exp(-pos_dist/l_pts);
                        // kernel_pos = pow(MAX(0., 1-pos_dist), power_pts);

                        // covariance of direction
                        view_dist = 1. - DOT(
                                train_x[stride*col+3], 
                                train_x[stride*col+4], 
                                train_x[stride*col+5],
                                x[stride*row+3],
                                x[stride*row+4],
                                x[stride*row+5]);
                        kernel_view = pow(MAX(1.-view_dist, 0), power_dir);
                        // kernel_view = exp(-view_dist/l_dir);
                        // kernel_view = pow(kernel_view, power_dir);

                        /// aggregated covariance
                        data[i] = kernel_pos * kernel_view;
                }
        }
}
}
