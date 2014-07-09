#include "kepler_solver_kernel_common.h"


void kepler_solver_kernel(
    const UINT ni,
    const REAL * restrict __im,
    const REAL * restrict __irx,
    const REAL * restrict __iry,
    const REAL * restrict __irz,
    const REAL * restrict __ie2,
    const REAL * restrict __ivx,
    const REAL * restrict __ivy,
    const REAL * restrict __ivz,
    const UINT nj,
    const REAL * restrict __jm,
    const REAL * restrict __jrx,
    const REAL * restrict __jry,
    const REAL * restrict __jrz,
    const REAL * restrict __je2,
    const REAL * restrict __jvx,
    const REAL * restrict __jvy,
    const REAL * restrict __jvz,
    const REAL dt,
    REAL * restrict __ir1x,
    REAL * restrict __ir1y,
    REAL * restrict __ir1z,
    REAL * restrict __iv1x,
    REAL * restrict __iv1y,
    REAL * restrict __iv1z)
{
    REAL ir1x[ni];
    REAL ir1y[ni];
    REAL ir1z[ni];
    REAL iv1x[ni];
    REAL iv1y[ni];
    REAL iv1z[ni];
    for (UINT i = 0; i < ni; ++i) {
        for (UINT j = i+1; j < nj; ++j) {
            kepler_solver_kernel_core(dt,
                               __im[i], __irx[i], __iry[i], __irz[i],
                               __ie2[i], __ivx[i], __ivy[i], __ivz[i],
                               __jm[j], __jrx[j], __jry[j], __jrz[j],
                               __je2[j], __jvx[j], __jvy[j], __jvz[j],
                               &ir1x[i], &ir1y[i], &ir1z[i],
                               &iv1x[i], &iv1y[i], &iv1z[i],
                               &ir1x[j], &ir1y[j], &ir1z[j],
                               &iv1x[j], &iv1y[j], &iv1z[j]);
        }
        __ir1x[i] = ir1x[i];
        __ir1y[i] = ir1y[i];
        __ir1z[i] = ir1z[i];
        __iv1x[i] = iv1x[i];
        __iv1y[i] = iv1y[i];
        __iv1z[i] = iv1z[i];
    }
}

