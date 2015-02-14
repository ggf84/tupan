#include "sakura_kernel_common.h"


void
sakura_kernel(
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
    REAL const dt,
    INT const flag,
    REAL __idrx[restrict],
    REAL __idry[restrict],
    REAL __idrz[restrict],
    REAL __idvx[restrict],
    REAL __idvy[restrict],
    REAL __idvz[restrict])
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
            REAL jm = __jm[j];
            REAL jrx = __jrx[j];
            REAL jry = __jry[j];
            REAL jrz = __jrz[j];
            REAL je2 = __je2[j];
            REAL jvx = __jvx[j];
            REAL jvy = __jvy[j];
            REAL jvz = __jvz[j];
            sakura_kernel_core(
                dt, flag,
                im, irx, iry, irz, ie2, ivx, ivy, ivz,
                jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                &idrx, &idry, &idrz, &idvx, &idvy, &idvz);
        }

        __idrx[i] = idrx;
        __idry[i] = idry;
        __idrz[i] = idrz;
        __idvx[i] = idvx;
        __idvy[i] = idvy;
        __idvz[i] = idvz;
    }
}

