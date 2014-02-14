#include "nreg_kernels_common.h"
#include "libtupan.h"


void nreg_Xkernel(
    const UINT ni,
    const REAL * restrict _im,
    const REAL * restrict _irx,
    const REAL * restrict _iry,
    const REAL * restrict _irz,
    const REAL * restrict _ie2,
    const REAL * restrict _ivx,
    const REAL * restrict _ivy,
    const REAL * restrict _ivz,
    const UINT nj,
    const REAL * restrict _jm,
    const REAL * restrict _jrx,
    const REAL * restrict _jry,
    const REAL * restrict _jrz,
    const REAL * restrict _je2,
    const REAL * restrict _jvx,
    const REAL * restrict _jvy,
    const REAL * restrict _jvz,
    const REAL dt,
    REAL * restrict _idrx,
    REAL * restrict _idry,
    REAL * restrict _idrz,
    REAL * restrict _iax,
    REAL * restrict _iay,
    REAL * restrict _iaz,
    REAL * restrict _iu)
{
    for (UINT i = 0; i < ni; ++i) {
        REAL im = _im[i];
        REAL irx = _irx[i];
        REAL iry = _iry[i];
        REAL irz = _irz[i];
        REAL ie2 = _ie2[i];
        REAL ivx = _ivx[i];
        REAL ivy = _ivy[i];
        REAL ivz = _ivz[i];
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
                              _jm[j], _jrx[j], _jry[j], _jrz[j],
                              _je2[j], _jvx[j], _jvy[j], _jvz[j],
                              &idrx, &idry, &idrz,
                              &iax, &iay, &iaz, &iu);
        }
        _idrx[i] = idrx;
        _idry[i] = idry;
        _idrz[i] = idrz;
        _iax[i] = iax;
        _iay[i] = iay;
        _iaz[i] = iaz;
        _iu[i] = im * iu;
    }
}


void nreg_Vkernel(
    const UINT ni,
    const REAL * restrict _im,
    const REAL * restrict _ivx,
    const REAL * restrict _ivy,
    const REAL * restrict _ivz,
    const REAL * restrict _iax,
    const REAL * restrict _iay,
    const REAL * restrict _iaz,
    const UINT nj,
    const REAL * restrict _jm,
    const REAL * restrict _jvx,
    const REAL * restrict _jvy,
    const REAL * restrict _jvz,
    const REAL * restrict _jax,
    const REAL * restrict _jay,
    const REAL * restrict _jaz,
    const REAL dt,
    REAL * restrict _idvx,
    REAL * restrict _idvy,
    REAL * restrict _idvz,
    REAL * restrict _ik)
{
    for (UINT i = 0; i < ni; ++i) {
        REAL im = _im[i];
        REAL ivx = _ivx[i];
        REAL ivy = _ivy[i];
        REAL ivz = _ivz[i];
        REAL iax = _iax[i];
        REAL iay = _iay[i];
        REAL iaz = _iaz[i];
        REAL idvx = 0;
        REAL idvy = 0;
        REAL idvz = 0;
        REAL ik = 0;
        for (UINT j = 0; j < nj; ++j) {
            nreg_Vkernel_core(dt,
                              im, ivx, ivy, ivz, iax, iay, iaz,
                              _jm[j], _jvx[j], _jvy[j], _jvz[j],
                              _jax[j], _jay[j], _jaz[j],
                              &idvx, &idvy, &idvz, &ik);
        }
        _idvx[i] = idvx;
        _idvy[i] = idvy;
        _idvz[i] = idvz;
        _ik[i] = im * ik;
    }
}

