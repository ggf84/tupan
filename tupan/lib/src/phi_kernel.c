#include "phi_kernel_common.h"


void phi_kernel(
    const UINT ni,
    const REAL * restrict __im,
    const REAL * restrict __irx,
    const REAL * restrict __iry,
    const REAL * restrict __irz,
    const REAL * restrict __ie2,
    const UINT nj,
    const REAL * restrict __jm,
    const REAL * restrict __jrx,
    const REAL * restrict __jry,
    const REAL * restrict __jrz,
    const REAL * restrict __je2,
    REAL * restrict __iphi)
{
    for (UINT i = 0; i < ni; ++i) {
        REAL im = __im[i];
        REAL irx = __irx[i];
        REAL iry = __iry[i];
        REAL irz = __irz[i];
        REAL ie2 = __ie2[i];
        REAL iphi = 0;

        for (UINT j = 0; j < nj; ++j) {
            REAL jm = __jm[j];
            REAL jrx = __jrx[j];
            REAL jry = __jry[j];
            REAL jrz = __jrz[j];
            REAL je2 = __je2[j];
            phi_kernel_core(
                im, irx, iry, irz, ie2,
                jm, jrx, jry, jrz, je2,
                &iphi);
        }

        __iphi[i] = iphi;
    }
}

