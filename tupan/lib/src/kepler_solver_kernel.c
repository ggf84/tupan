#include "kepler_solver_kernel_common.h"
#include "libtupan.h"


void kepler_solver_kernel(
    const REAL * restrict _im,
    const REAL * restrict _irx,
    const REAL * restrict _iry,
    const REAL * restrict _irz,
    const REAL * restrict _ie2,
    const REAL * restrict _ivx,
    const REAL * restrict _ivy,
    const REAL * restrict _ivz,
    const REAL dt,
    REAL * restrict _ir1x,
    REAL * restrict _ir1y,
    REAL * restrict _ir1z,
    REAL * restrict _iv1x,
    REAL * restrict _iv1y,
    REAL * restrict _iv1z)
{
    REAL ir1x, ir1y, ir1z;
    REAL iv1x, iv1y, iv1z;
    REAL jr1x, jr1y, jr1z;
    REAL jv1x, jv1y, jv1z;

    kepler_solver_kernel_core(dt,
                              _im[0], _irx[0], _iry[0], _irz[0],
                              _ie2[0], _ivx[0], _ivy[0], _ivz[0],
                              _im[1], _irx[1], _iry[1], _irz[1],
                              _ie2[1], _ivx[1], _ivy[1], _ivz[1],
                              &ir1x, &ir1y, &ir1z,
                              &iv1x, &iv1y, &iv1z,
                              &jr1x, &jr1y, &jr1z,
                              &jv1x, &jv1y, &jv1z);

    _ir1x[0] = ir1x;
    _ir1y[0] = ir1y;
    _ir1z[0] = ir1z;
    _iv1x[0] = iv1x;
    _iv1y[0] = iv1y;
    _iv1z[0] = iv1z;
    _ir1x[1] = jr1x;
    _ir1y[1] = jr1y;
    _ir1z[1] = jr1z;
    _iv1x[1] = jv1x;
    _iv1y[1] = jv1y;
    _iv1z[1] = jv1z;
}

