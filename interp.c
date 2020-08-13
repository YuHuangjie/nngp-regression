#include <math.h>
#include <stddef.h>

#define MIN(x, y) (x) < (y) ? (x) : (y)

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
        int idx_x[2], idx_y[2];
        double weight_x[2], weight_y[2];

#pragma omp parallel for num_threads(39)
        for (int i = 0; i < sz_yp; ++i) {
                get_interp_idx_weight(x, sz_x, y, sz_y, xp, yp[i], idx_x, idx_y,
                        weight_x, weight_y);
                out[i] = 0.;
                out[i] += z[idx_x[0]*sz_y+idx_y[0]]*weight_x[0]*weight_y[0];
                out[i] += z[idx_x[0]*sz_y+idx_y[1]]*weight_x[0]*weight_y[1];
                out[i] += z[idx_x[1]*sz_y+idx_y[0]]*weight_x[1]*weight_y[0];
                out[i] += z[idx_x[1]*sz_y+idx_y[1]]*weight_x[1]*weight_y[1];
        }
}
}
