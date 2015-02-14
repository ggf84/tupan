#include "acc_kernel_common.h"


void
acc_kernel(
    UINT const ni,
    REAL const __im[restrict],
    REAL const __irx[restrict],
    REAL const __iry[restrict],
    REAL const __irz[restrict],
    REAL const __ie2[restrict],
    UINT const nj,
    REAL const __jm[restrict],
    REAL const __jrx[restrict],
    REAL const __jry[restrict],
    REAL const __jrz[restrict],
    REAL const __je2[restrict],
    REAL __iax[restrict],
    REAL __iay[restrict],
    REAL __iaz[restrict])
{
    for (UINT i = 0; i < ni; ++i) {
        REAL im = __im[i];
        REAL irx = __irx[i];
        REAL iry = __iry[i];
        REAL irz = __irz[i];
        REAL ie2 = __ie2[i];
        REAL iax = 0;
        REAL iay = 0;
        REAL iaz = 0;

        for (UINT j = 0; j < nj; ++j) {
            REAL jm = __jm[j];
            REAL jrx = __jrx[j];
            REAL jry = __jry[j];
            REAL jrz = __jrz[j];
            REAL je2 = __je2[j];
            acc_kernel_core(
                im, irx, iry, irz, ie2,
                jm, jrx, jry, jrz, je2,
                &iax, &iay, &iaz);
        }

        __iax[i] = iax;
        __iay[i] = iay;
        __iaz[i] = iaz;
    }
}

