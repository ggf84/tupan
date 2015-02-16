#include "kepler_solver_kernel_common.h"


void
kepler_solver_kernel(
    uint_t const ni,
    real_t const __im[restrict],
    real_t const __irx[restrict],
    real_t const __iry[restrict],
    real_t const __irz[restrict],
    real_t const __ie2[restrict],
    real_t const __ivx[restrict],
    real_t const __ivy[restrict],
    real_t const __ivz[restrict],
    uint_t const nj,
    real_t const __jm[restrict],
    real_t const __jrx[restrict],
    real_t const __jry[restrict],
    real_t const __jrz[restrict],
    real_t const __je2[restrict],
    real_t const __jvx[restrict],
    real_t const __jvy[restrict],
    real_t const __jvz[restrict],
    real_t const dt,
    real_t __ir1x[restrict],
    real_t __ir1y[restrict],
    real_t __ir1z[restrict],
    real_t __iv1x[restrict],
    real_t __iv1y[restrict],
    real_t __iv1z[restrict],
    real_t __jr1x[restrict],
    real_t __jr1y[restrict],
    real_t __jr1z[restrict],
    real_t __jv1x[restrict],
    real_t __jv1y[restrict],
    real_t __jv1z[restrict])
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

