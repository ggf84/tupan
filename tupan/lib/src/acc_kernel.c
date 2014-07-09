#include "acc_kernel_common.h"


void acc_kernel(
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
    REAL * restrict __iax,
    REAL * restrict __iay,
    REAL * restrict __iaz)
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
            acc_kernel_core(im, irx, iry, irz, ie2,
                            __jm[j], __jrx[j], __jry[j], __jrz[j], __je2[j],
                            &iax, &iay, &iaz);
        }
        __iax[i] = iax;
        __iay[i] = iay;
        __iaz[i] = iaz;
    }
}

