#include "kepler_solver_kernel_common.h"


void
kepler_solver_kernel(
    UINT const ni,
    REAL const __im[restrict],
    REAL const __irx[restrict],
    REAL const __iry[restrict],
    REAL const __irz[restrict],
    REAL const __ie2[restrict],
    REAL const __ivx[restrict],
    REAL const __ivy[restrict],
    REAL const __ivz[restrict],
    UINT const nj,
    REAL const __jm[restrict],
    REAL const __jrx[restrict],
    REAL const __jry[restrict],
    REAL const __jrz[restrict],
    REAL const __je2[restrict],
    REAL const __jvx[restrict],
    REAL const __jvy[restrict],
    REAL const __jvz[restrict],
    REAL const dt,
    REAL __ir1x[restrict],
    REAL __ir1y[restrict],
    REAL __ir1z[restrict],
    REAL __iv1x[restrict],
    REAL __iv1y[restrict],
    REAL __iv1z[restrict],
    REAL __jr1x[restrict],
    REAL __jr1y[restrict],
    REAL __jr1z[restrict],
    REAL __jv1x[restrict],
    REAL __jv1y[restrict],
    REAL __jv1z[restrict])
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

