#include "acc_jerk_kernel_common.h"


void acc_jerk_kernel(
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
    REAL * restrict __iax,
    REAL * restrict __iay,
    REAL * restrict __iaz,
    REAL * restrict __ijx,
    REAL * restrict __ijy,
    REAL * restrict __ijz)
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

