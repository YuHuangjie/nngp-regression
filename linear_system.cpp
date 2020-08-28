#include <stdio.h>
#include <chrono>
#include "mkl_rci.h"
#include "mkl_blas.h"
#include "mkl_spblas.h"
#include "mkl_service.h"

using namespace std;


void m_dot_v(const MKL_INT *__restrict ia, const MKL_INT *__restrict ja, 
        const double *__restrict a, MKL_INT n, const double *__restrict x, 
        double *__restrict y)
{
#pragma omp parallel for num_threads(40)
        for (MKL_INT i = 0; i < n; i++) {
                MKL_INT start = ia[i], end = ia[i+1];
                volatile double tmp = 0.;
#pragma GCC ivdep
                for (MKL_INT k = start; k < end; k++) {
                        tmp += a[k] * x[ja[k]];
                }
                y[i] = tmp;
        }
}

void _solve_system(const MKL_INT *ia, const MKL_INT *ja, const double *a,
        MKL_INT n, const double *b, double *x, MKL_INT *itercount)
{
        MKL_INT rci_request;
        MKL_INT ipar[128];
        double dpar[128];
        double *tmp = new double[n*4];
        double *temp = new double[n];
        chrono::steady_clock::time_point start, end;

        for (MKL_INT i = 0; i < n; i++)
                x[i] = 0.;

        dcg_init(&n, x, b, &rci_request, ipar, dpar, tmp);
        if (rci_request != 0)
                goto failure;

        ipar[7] = 0;
        ipar[8] = 1;
        ipar[9] = 1;
        dpar[0] = 1e-7;
        dcg_check (&n, x, b, &rci_request, ipar, dpar, tmp);
        if (rci_request != 0)
                goto failure;
        
        start = chrono::steady_clock::now();
rci:    dcg(&n, x, b, &rci_request, ipar, dpar, tmp);
        if (rci_request == 0)
                goto getsln;
        if (rci_request == 1) {
                // mkl_sparse_d_mv(transA, 1.0, csrA, descrA, tmp, 0.0, &tmp[n]);
                m_dot_v(ia, ja, a, n, tmp, &tmp[n]);
                goto rci;
        }
        if (rci_request == 2)
        {
                if (dpar[4] > dpar[5] && dpar[5] > 0)
                        goto getsln;
                else
                        goto rci;
        }
        goto failure;
getsln: dcg_get (&n, x, b, &rci_request, ipar, dpar, tmp, itercount);
        end = chrono::steady_clock::now();

        printf("%ld ms\n", chrono::duration_cast<chrono::milliseconds>(end-start).count());
        printf ("The system has been solved\n");
        printf ("Number of iterations: %lld\n", *itercount);

        return;

failure:printf ("This example FAILED as the solver has returned the ERROR code %lld", rci_request);
}

extern "C" {
void solve_system(
        const int64_t *ia, 
        const int64_t *ja, 
        const double *a, 
        int n, 
        const double *mrhs, 
        int nrhs, 
        double *x, 
        int64_t *itercount)
{
        const MKL_INT *llia = reinterpret_cast<const MKL_INT*>(ia);
        const MKL_INT *llja = reinterpret_cast<const MKL_INT*>(ja);
        MKL_INT *lliter = reinterpret_cast<MKL_INT*>(itercount);

        for (int i = 0; i < nrhs; i++)
                _solve_system(llia, llja, a, n, &mrhs[i*n], &x[i*n], &lliter[i]);
}

}