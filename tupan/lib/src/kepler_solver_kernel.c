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
    REAL * restrict __iv1z,
    REAL * restrict __jr1x,
    REAL * restrict __jr1y,
    REAL * restrict __jr1z,
    REAL * restrict __jv1x,
    REAL * restrict __jv1y,
    REAL * restrict __jv1z)
{
    kepler_solver_kernel_core(dt,
                       __im[0], __irx[0], __iry[0], __irz[0],
                       __ie2[0], __ivx[0], __ivy[0], __ivz[0],
                       __jm[0], __jrx[0], __jry[0], __jrz[0],
                       __je2[0], __jvx[0], __jvy[0], __jvz[0],
                       __ir1x, __ir1y, __ir1z,
                       __iv1x, __iv1y, __iv1z,
                       __jr1x, __jr1y, __jr1z,
                       __jv1x, __jv1y, __jv1z);
}

