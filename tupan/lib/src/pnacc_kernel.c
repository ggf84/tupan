#include "pnacc_kernel_common.h"


void
pnacc_kernel(
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
    UINT const order,
    REAL const inv1,
    REAL const inv2,
    REAL const inv3,
    REAL const inv4,
    REAL const inv5,
    REAL const inv6,
    REAL const inv7,
    REAL __ipnax[restrict],
    REAL __ipnay[restrict],
    REAL __ipnaz[restrict])
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

