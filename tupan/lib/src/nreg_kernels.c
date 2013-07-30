#include "nreg_kernels_common.h"


inline void nreg_Xkernel(
    const unsigned int ni,
    const REAL *_im,
    const REAL *_irx,
    const REAL *_iry,
    const REAL *_irz,
    const REAL *_ie2,
    const REAL *_ivx,
    const REAL *_ivy,
    const REAL *_ivz,
    const unsigned int nj,
    const REAL *_jm,
    const REAL *_jrx,
    const REAL *_jry,
    const REAL *_jrz,
    const REAL *_je2,
    const REAL *_jvx,
    const REAL *_jvy,
    const REAL *_jvz,
    const REAL dt,
    REAL *_idrx,
    REAL *_idry,
    REAL *_idrz,
    REAL *_iax,
    REAL *_iay,
    REAL *_iaz,
    REAL *_iu)
{
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
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
        for (j = 0; j < nj; ++j) {
            REAL jm = _jm[j];
            REAL jrx = _jrx[j];
            REAL jry = _jry[j];
            REAL jrz = _jrz[j];
            REAL je2 = _je2[j];
            REAL jvx = _jvx[j];
            REAL jvy = _jvy[j];
            REAL jvz = _jvz[j];
            nreg_Xkernel_core(dt,
                              im, irx, iry, irz, ie2, ivx, ivy, ivz,
                              jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                              &idrx, &idry, &idrz,
                              &iax, &iay, &iaz, &iu);
        }
        _idrx[i] = idrx;
        _idry[i] = idry;
        _idrz[i] = idrz;
        _iax[i] = iax;
        _iay[i] = iay;
        _iaz[i] = iaz;
        _iu[i] = iu;
    }
}


inline void nreg_Vkernel(
    const unsigned int ni,
    const REAL *_im,
    const REAL *_ivx,
    const REAL *_ivy,
    const REAL *_ivz,
    const REAL *_iax,
    const REAL *_iay,
    const REAL *_iaz,
    const unsigned int nj,
    const REAL *_jm,
    const REAL *_jvx,
    const REAL *_jvy,
    const REAL *_jvz,
    const REAL *_jax,
    const REAL *_jay,
    const REAL *_jaz,
    const REAL dt,
    REAL *_idvx,
    REAL *_idvy,
    REAL *_idvz,
    REAL *_ik)
{
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
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
        for (j = 0; j < nj; ++j) {
            REAL jm = _jm[j];
            REAL jvx = _jvx[j];
            REAL jvy = _jvy[j];
            REAL jvz = _jvz[j];
            REAL jax = _jax[j];
            REAL jay = _jay[j];
            REAL jaz = _jaz[j];
            nreg_Vkernel_core(dt,
                              im, ivx, ivy, ivz, iax, iay, iaz,
                              jm, jvx, jvy, jvz, jax, jay, jaz,
                              &idvx, &idvy, &idvz, &ik);
        }
        _idvx[i] = idvx;
        _idvy[i] = idvy;
        _idvz[i] = idvz;
        _ik[i] = ik;
    }
}

