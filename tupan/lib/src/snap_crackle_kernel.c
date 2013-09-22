#include "snap_crackle_kernel_common.h"


inline void snap_crackle_kernel(
    const UINT ni,
    const REAL *_im,
    const REAL *_irx,
    const REAL *_iry,
    const REAL *_irz,
    const REAL *_ie2,
    const REAL *_ivx,
    const REAL *_ivy,
    const REAL *_ivz,
    const REAL *_iax,
    const REAL *_iay,
    const REAL *_iaz,
    const REAL *_ijx,
    const REAL *_ijy,
    const REAL *_ijz,
    const UINT nj,
    const REAL *_jm,
    const REAL *_jrx,
    const REAL *_jry,
    const REAL *_jrz,
    const REAL *_je2,
    const REAL *_jvx,
    const REAL *_jvy,
    const REAL *_jvz,
    const REAL *_jax,
    const REAL *_jay,
    const REAL *_jaz,
    const REAL *_jjx,
    const REAL *_jjy,
    const REAL *_jjz,
    REAL *_isx,
    REAL *_isy,
    REAL *_isz,
    REAL *_icx,
    REAL *_icy,
    REAL *_icz)
{
    UINT i, j;
    for (i = 0; i < ni; ++i) {
        REAL im = _im[i];
        REAL irx = _irx[i];
        REAL iry = _iry[i];
        REAL irz = _irz[i];
        REAL ie2 = _ie2[i];
        REAL ivx = _ivx[i];
        REAL ivy = _ivy[i];
        REAL ivz = _ivz[i];
        REAL iax = _iax[i];
        REAL iay = _iay[i];
        REAL iaz = _iaz[i];
        REAL ijx = _ijx[i];
        REAL ijy = _ijy[i];
        REAL ijz = _ijz[i];
        REAL isx = 0;
        REAL isy = 0;
        REAL isz = 0;
        REAL icx = 0;
        REAL icy = 0;
        REAL icz = 0;
        for (j = 0; j < nj; ++j) {
            REAL jm = _jm[j];
            REAL jrx = _jrx[j];
            REAL jry = _jry[j];
            REAL jrz = _jrz[j];
            REAL je2 = _je2[j];
            REAL jvx = _jvx[j];
            REAL jvy = _jvy[j];
            REAL jvz = _jvz[j];
            REAL jax = _jax[j];
            REAL jay = _jay[j];
            REAL jaz = _jaz[j];
            REAL jjx = _jjx[j];
            REAL jjy = _jjy[j];
            REAL jjz = _jjz[j];
            snap_crackle_kernel_core(im, irx, iry, irz,
                                     ie2, ivx, ivy, ivz,
                                     iax, iay, iaz, ijx, ijy, ijz,
                                     jm, jrx, jry, jrz,
                                     je2, jvx, jvy, jvz,
                                     jax, jay, jaz, jjx, jjy, jjz,
                                     &isx, &isy, &isz,
                                     &icx, &icy, &icz);
        }
        _isx[i] = isx;
        _isy[i] = isy;
        _isz[i] = isz;
        _icx[i] = icx;
        _icy[i] = icy;
        _icz[i] = icz;
    }
}

