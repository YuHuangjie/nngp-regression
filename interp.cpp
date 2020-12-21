#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

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

static double interp_lin_2d(
        const double *x, 
        size_t sz_x, // no. elements in x
        const double *y, 
        size_t sz_y, // no. elements in y
        const double *z, 
        double xp, 
        double yp)
{
        int idx_x[2], idx_y[2];
        double weight_x[2], weight_y[2];
        double out = 0.;

        get_interp_idx_weight(x, sz_x, y, sz_y, xp, yp, idx_x, idx_y,
                weight_x, weight_y);
        out += z[idx_x[0]*sz_y+idx_y[0]]*weight_x[0]*weight_y[0];
        out += z[idx_x[0]*sz_y+idx_y[1]]*weight_x[0]*weight_y[1];
        out += z[idx_x[1]*sz_y+idx_y[0]]*weight_x[1]*weight_y[0];
        out += z[idx_x[1]*sz_y+idx_y[1]]*weight_x[1]*weight_y[1];
        return out;
}

extern "C" {
double recursive_kernel(
        const double *x, 
        size_t sz_x, // no. elements in x
        const double *y, 
        size_t sz_y, // no. elements in y
        const double *z, 
        float *yp_in_out, 
        size_t sz_yp, 
        size_t depth,
        double weight_var,
        double bias_var,
        const double *layer_qaa)
{
        double cov0 = bias_var;
        {
                double corr, xp, yp;

                for (int d = 0; d < depth; ++d) {
                        xp = layer_qaa[d];
                        yp = corr = cov0 / layer_qaa[d];
                        cov0 = interp_lin_2d(x, sz_x, y, sz_y, z, xp, yp);
                        cov0 = cov0 * weight_var + bias_var;
                }
        }

        // recursively calculate covariance
#pragma omp parallel for num_threads(40)
        for (size_t i = 0; i < sz_yp; ++i) {
                double cov = yp_in_out[i] * weight_var + bias_var;
                double corr, xp, yp;

                for (int d = 0; d < depth; d++) {
                        xp = layer_qaa[d];
                        yp = corr = cov / layer_qaa[d];
                        cov = interp_lin_2d(x, sz_x, y, sz_y, z, xp, yp);
                        cov = cov * weight_var + bias_var;
                }
                cov -= cov0;
                yp_in_out[i] = cov;
        }
        return cov0;
}

void lin_interp(
        const float *x, 
        size_t sz_x,
        const float *y, 
        float *xp_in_out, 
        size_t sz_xp)
{
        float spacing = x[1] - x[0];

#pragma omp parallel for num_threads(40)
        for (size_t i = 0; i < sz_xp; i++) {
                float xp = xp_in_out[i];
                float grid = (xp - x[0]) / spacing;
                size_t ind1 = (size_t)grid;
                size_t ind2 = MIN(ind1 + 1, sz_x - 1);
                float weight1 = 1. - abs(xp - x[ind1]) / spacing;
                float weight2 = 1. - abs(xp - x[ind2]) / spacing;
                float weight_sum = weight1 + weight2;
                weight1 /= weight_sum;
                weight2 /= weight_sum;
                xp_in_out[i] = y[ind1] * weight1 + y[ind2] * weight2;
        }
}
}