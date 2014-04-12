#include "kepler_solver_kernel_common.h"
#include "libtupan.h"


void kepler_solver_kernel(
    const UINT ni,
    const REAL * restrict _im,
    const REAL * restrict _irx,
    const REAL * restrict _iry,
    const REAL * restrict _irz,
    const REAL * restrict _ie2,
    const REAL * restrict _ivx,
    const REAL * restrict _ivy,
    const REAL * restrict _ivz,
    const UINT nj,
    const REAL * restrict _jm,
    const REAL * restrict _jrx,
    const REAL * restrict _jry,
    const REAL * restrict _jrz,
    const REAL * restrict _je2,
    const REAL * restrict _jvx,
    const REAL * restrict _jvy,
    const REAL * restrict _jvz,
    const REAL dt,
    REAL * restrict _ir1x,
    REAL * restrict _ir1y,
    REAL * restrict _ir1z,
    REAL * restrict _iv1x,
    REAL * restrict _iv1y,
    REAL * restrict _iv1z)
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
                               _im[i], _irx[i], _iry[i], _irz[i],
                               _ie2[i], _ivx[i], _ivy[i], _ivz[i],
                               _jm[j], _jrx[j], _jry[j], _jrz[j],
                               _je2[j], _jvx[j], _jvy[j], _jvz[j],
                               &ir1x[i], &ir1y[i], &ir1z[i],
                               &iv1x[i], &iv1y[i], &iv1z[i],
                               &ir1x[j], &ir1y[j], &ir1z[j],
                               &iv1x[j], &iv1y[j], &iv1z[j]);
        }
        _ir1x[i] = ir1x[i];
        _ir1y[i] = ir1y[i];
        _ir1z[i] = ir1z[i];
        _iv1x[i] = iv1x[i];
        _iv1y[i] = iv1y[i];
        _iv1z[i] = iv1z[i];
    }
}

