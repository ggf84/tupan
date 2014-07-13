#include "pnacc_kernel_common.h"


void pnacc_kernel(
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
    UINT order,
    const REAL inv1,
    const REAL inv2,
    const REAL inv3,
    const REAL inv4,
    const REAL inv5,
    const REAL inv6,
    const REAL inv7,
    REAL * restrict __ipnax,
    REAL * restrict __ipnay,
    REAL * restrict __ipnaz)
{
    CLIGHT clight = CLIGHT_Init(order, inv1, inv2, inv3,
                                inv4, inv5, inv6, inv7);
    for (UINT i = 0; i < ni; ++i) {
        REAL im = __im[i];
        REAL irx = __irx[i];
        REAL iry = __iry[i];
        REAL irz = __irz[i];
        REAL ie2 = __ie2[i];
        REAL ivx = __ivx[i];
        REAL ivy = __ivy[i];
        REAL ivz = __ivz[i];
        REAL ipnax = 0;
        REAL ipnay = 0;
        REAL ipnaz = 0;

        for (UINT j = 0; j < nj; ++j) {
            REAL jm = __jm[j];
            REAL jrx = __jrx[j];
            REAL jry = __jry[j];
            REAL jrz = __jrz[j];
            REAL je2 = __je2[j];
            REAL jvx = __jvx[j];
            REAL jvy = __jvy[j];
            REAL jvz = __jvz[j];
            pnacc_kernel_core(
                im, irx, iry, irz, ie2, ivx, ivy, ivz,
                jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                clight,
                &ipnax, &ipnay, &ipnaz);
        }

        __ipnax[i] = ipnax;
        __ipnay[i] = ipnay;
        __ipnaz[i] = ipnaz;
    }
}

