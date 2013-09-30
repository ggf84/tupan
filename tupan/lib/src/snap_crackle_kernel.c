#include "snap_crackle_kernel_common.h"
#include "libtupan.h"


void snap_crackle_kernel(
    const UINT ni,
    const REAL * restrict _im,
    const REAL * restrict _irx,
    const REAL * restrict _iry,
    const REAL * restrict _irz,
    const REAL * restrict _ie2,
    const REAL * restrict _ivx,
    const REAL * restrict _ivy,
    const REAL * restrict _ivz,
    const REAL * restrict _iax,
    const REAL * restrict _iay,
    const REAL * restrict _iaz,
    const REAL * restrict _ijx,
    const REAL * restrict _ijy,
    const REAL * restrict _ijz,
    const UINT nj,
    const REAL * restrict _jm,
    const REAL * restrict _jrx,
    const REAL * restrict _jry,
    const REAL * restrict _jrz,
    const REAL * restrict _je2,
    const REAL * restrict _jvx,
    const REAL * restrict _jvy,
    const REAL * restrict _jvz,
    const REAL * restrict _jax,
    const REAL * restrict _jay,
    const REAL * restrict _jaz,
    const REAL * restrict _jjx,
    const REAL * restrict _jjy,
    const REAL * restrict _jjz,
    REAL * restrict _isx,
    REAL * restrict _isy,
    REAL * restrict _isz,
    REAL * restrict _icx,
    REAL * restrict _icy,
    REAL * restrict _icz)
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
        for (UINT j = 0; j < nj; ++j) {
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

