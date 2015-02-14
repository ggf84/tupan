#include "nreg_kernels_common.h"


void
nreg_Xkernel(
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
    REAL __idrx[restrict],
    REAL __idry[restrict],
    REAL __idrz[restrict],
    REAL __iax[restrict],
    REAL __iay[restrict],
    REAL __iaz[restrict],
    REAL __iu[restrict])
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
        REAL iax = 0;
        REAL iay = 0;
        REAL iaz = 0;
        REAL iu = 0;

        for (UINT j = 0; j < nj; ++j) {
            REAL jm = __jm[j];
            REAL jrx = __jrx[j];
            REAL jry = __jry[j];
            REAL jrz = __jrz[j];
            REAL je2 = __je2[j];
            REAL jvx = __jvx[j];
            REAL jvy = __jvy[j];
            REAL jvz = __jvz[j];
            nreg_Xkernel_core(
                dt,
                im, irx, iry, irz, ie2, ivx, ivy, ivz,
                jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                &idrx, &idry, &idrz, &iax, &iay, &iaz, &iu);
        }

        __idrx[i] = idrx;
        __idry[i] = idry;
        __idrz[i] = idrz;
        __iax[i] = iax;
        __iay[i] = iay;
        __iaz[i] = iaz;
        __iu[i] = im * iu;
    }
}


void
nreg_Vkernel(
    UINT const ni,
    REAL const __im[restrict],
    REAL const __ivx[restrict],
    REAL const __ivy[restrict],
    REAL const __ivz[restrict],
    REAL const __iax[restrict],
    REAL const __iay[restrict],
    REAL const __iaz[restrict],
    UINT const nj,
    REAL const __jm[restrict],
    REAL const __jvx[restrict],
    REAL const __jvy[restrict],
    REAL const __jvz[restrict],
    REAL const __jax[restrict],
    REAL const __jay[restrict],
    REAL const __jaz[restrict],
    REAL const dt,
    REAL __idvx[restrict],
    REAL __idvy[restrict],
    REAL __idvz[restrict],
    REAL __ik[restrict])
{
    for (UINT i = 0; i < ni; ++i) {
        REAL im = __im[i];
        REAL ivx = __ivx[i];
        REAL ivy = __ivy[i];
        REAL ivz = __ivz[i];
        REAL iax = __iax[i];
        REAL iay = __iay[i];
        REAL iaz = __iaz[i];
        REAL idvx = 0;
        REAL idvy = 0;
        REAL idvz = 0;
        REAL ik = 0;

        for (UINT j = 0; j < nj; ++j) {
            REAL jm = __jm[j];
            REAL jvx = __jvx[j];
            REAL jvy = __jvy[j];
            REAL jvz = __jvz[j];
            REAL jax = __jax[j];
            REAL jay = __jay[j];
            REAL jaz = __jaz[j];
            nreg_Vkernel_core(
                dt,
                im, ivx, ivy, ivz, iax, iay, iaz,
                jm, jvx, jvy, jvz, jax, jay, jaz,
                &idvx, &idvy, &idvz, &ik);
        }

        __idvx[i] = idvx;
        __idvy[i] = idvy;
        __idvz[i] = idvz;
        __ik[i] = im * ik;
    }
}

