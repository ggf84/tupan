#include "sakura_kernel_common.h"


void sakura_kernel(
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
    const REAL dt,
    const INT flag,
    REAL * restrict __idrx,
    REAL * restrict __idry,
    REAL * restrict __idrz,
    REAL * restrict __idvx,
    REAL * restrict __idvy,
    REAL * restrict __idvz)
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
        REAL idrx = 0;
        REAL idry = 0;
        REAL idrz = 0;
        REAL idvx = 0;
        REAL idvy = 0;
        REAL idvz = 0;
        for (UINT j = 0; j < nj; ++j) {
            sakura_kernel_core(dt, flag,
                               im, irx, iry, irz, ie2, ivx, ivy, ivz,
                               __jm[j], __jrx[j], __jry[j], __jrz[j],
                               __je2[j], __jvx[j], __jvy[j], __jvz[j],
                               &idrx, &idry, &idrz,
                               &idvx, &idvy, &idvz);
        }
        __idrx[i] = idrx;
        __idry[i] = idry;
        __idrz[i] = idrz;
        __idvx[i] = idvx;
        __idvy[i] = idvy;
        __idvz[i] = idvz;
    }
}

