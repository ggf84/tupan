#include "acc_jerk_kernel_common.h"


void
acc_jerk_kernel(
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
    REAL __iax[restrict],
    REAL __iay[restrict],
    REAL __iaz[restrict],
    REAL __ijx[restrict],
    REAL __ijy[restrict],
    REAL __ijz[restrict])
{
    for (UINT i = 0; i < ni; ++i) {
        REAL im = __im[i];
        REAL irx = __irx[i];
        REAL iry = __iry[i];
        REAL irz = __irz[i];
        REAL ie2 = __ie2[i];
        REAL ivx = __ivx[i];
        REAL ivy = __ivy[i];
        REAL ivz = __ivz[i];
        REAL iax = 0;
        REAL iay = 0;
        REAL iaz = 0;
        REAL ijx = 0;
        REAL ijy = 0;
        REAL ijz = 0;

        for (UINT j = 0; j < nj; ++j) {
            REAL jm = __jm[j];
            REAL jrx = __jrx[j];
            REAL jry = __jry[j];
            REAL jrz = __jrz[j];
            REAL je2 = __je2[j];
            REAL jvx = __jvx[j];
            REAL jvy = __jvy[j];
            REAL jvz = __jvz[j];
            acc_jerk_kernel_core(
                im, irx, iry, irz, ie2, ivx, ivy, ivz,
                jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                &iax, &iay, &iaz, &ijx, &ijy, &ijz);
        }

        __iax[i] = iax;
        __iay[i] = iay;
        __iaz[i] = iaz;
        __ijx[i] = ijx;
        __ijy[i] = ijy;
        __ijz[i] = ijz;
    }
}

