#include "phi_kernel_common.h"


void
phi_kernel(
    uint_t const ni,
    real_t const __im[restrict],
    real_t const __irx[restrict],
    real_t const __iry[restrict],
    real_t const __irz[restrict],
    real_t const __ie2[restrict],
    uint_t const nj,
    real_t const __jm[restrict],
    real_t const __jrx[restrict],
    real_t const __jry[restrict],
    real_t const __jrz[restrict],
    real_t const __je2[restrict],
    real_t __iphi[restrict])
{
    for (uint_t i = 0; i < ni; ++i) {
        real_t im = __im[i];
        real_t irx = __irx[i];
        real_t iry = __iry[i];
        real_t irz = __irz[i];
        real_t ie2 = __ie2[i];
        real_t iphi = 0;

        for (uint_t j = 0; j < nj; ++j) {
            real_t jm = __jm[j];
            real_t jrx = __jrx[j];
            real_t jry = __jry[j];
            real_t jrz = __jrz[j];
            real_t je2 = __je2[j];
            phi_kernel_core(
                im, irx, iry, irz, ie2,
                jm, jrx, jry, jrz, je2,
                &iphi);
        }

        __iphi[i] = iphi;
    }
}

