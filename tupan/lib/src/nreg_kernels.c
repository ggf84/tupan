#include "nreg_kernels_common.h"


void nreg_Xkernel(
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
    REAL * restrict __idrx,
    REAL * restrict __idry,
    REAL * restrict __idrz,
    REAL * restrict __iax,
    REAL * restrict __iay,
    REAL * restrict __iaz,
    REAL * restrict __iu)
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
            nreg_Xkernel_core(dt,
                              im, irx, iry, irz, ie2, ivx, ivy, ivz,
                              __jm[j], __jrx[j], __jry[j], __jrz[j],
                              __je2[j], __jvx[j], __jvy[j], __jvz[j],
                              &idrx, &idry, &idrz,
                              &iax, &iay, &iaz, &iu);
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


void nreg_Vkernel(
    const UINT ni,
    const REAL * restrict __im,
    const REAL * restrict __ivx,
    const REAL * restrict __ivy,
    const REAL * restrict __ivz,
    const REAL * restrict __iax,
    const REAL * restrict __iay,
    const REAL * restrict __iaz,
    const UINT nj,
    const REAL * restrict __jm,
    const REAL * restrict __jvx,
    const REAL * restrict __jvy,
    const REAL * restrict __jvz,
    const REAL * restrict __jax,
    const REAL * restrict __jay,
    const REAL * restrict __jaz,
    const REAL dt,
    REAL * restrict __idvx,
    REAL * restrict __idvy,
    REAL * restrict __idvz,
    REAL * restrict __ik)
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
            nreg_Vkernel_core(dt,
                              im, ivx, ivy, ivz, iax, iay, iaz,
                              __jm[j], __jvx[j], __jvy[j], __jvz[j],
                              __jax[j], __jay[j], __jaz[j],
                              &idvx, &idvy, &idvz, &ik);
        }
        __idvx[i] = idvx;
        __idvy[i] = idvy;
        __idvz[i] = idvz;
        __ik[i] = im * ik;
    }
}

