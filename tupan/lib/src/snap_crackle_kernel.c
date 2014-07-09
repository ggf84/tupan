#include "snap_crackle_kernel_common.h"


void snap_crackle_kernel(
    const UINT ni,
    const REAL * restrict __im,
    const REAL * restrict __irx,
    const REAL * restrict __iry,
    const REAL * restrict __irz,
    const REAL * restrict __ie2,
    const REAL * restrict __ivx,
    const REAL * restrict __ivy,
    const REAL * restrict __ivz,
    const REAL * restrict __iax,
    const REAL * restrict __iay,
    const REAL * restrict __iaz,
    const REAL * restrict __ijx,
    const REAL * restrict __ijy,
    const REAL * restrict __ijz,
    const UINT nj,
    const REAL * restrict __jm,
    const REAL * restrict __jrx,
    const REAL * restrict __jry,
    const REAL * restrict __jrz,
    const REAL * restrict __je2,
    const REAL * restrict __jvx,
    const REAL * restrict __jvy,
    const REAL * restrict __jvz,
    const REAL * restrict __jax,
    const REAL * restrict __jay,
    const REAL * restrict __jaz,
    const REAL * restrict __jjx,
    const REAL * restrict __jjy,
    const REAL * restrict __jjz,
    REAL * restrict __isx,
    REAL * restrict __isy,
    REAL * restrict __isz,
    REAL * restrict __icx,
    REAL * restrict __icy,
    REAL * restrict __icz)
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
        REAL iax = __iax[i];
        REAL iay = __iay[i];
        REAL iaz = __iaz[i];
        REAL ijx = __ijx[i];
        REAL ijy = __ijy[i];
        REAL ijz = __ijz[i];
        REAL isx = 0;
        REAL isy = 0;
        REAL isz = 0;
        REAL icx = 0;
        REAL icy = 0;
        REAL icz = 0;
        for (UINT j = 0; j < nj; ++j) {
            snap_crackle_kernel_core(im, irx, iry, irz,
                                     ie2, ivx, ivy, ivz,
                                     iax, iay, iaz, ijx, ijy, ijz,
                                     __jm[j], __jrx[j], __jry[j], __jrz[j],
                                     __je2[j], __jvx[j], __jvy[j], __jvz[j],
                                     __jax[j], __jay[j], __jaz[j],
                                     __jjx[j], __jjy[j], __jjz[j],
                                     &isx, &isy, &isz,
                                     &icx, &icy, &icz);
        }
        __isx[i] = isx;
        __isy[i] = isy;
        __isz[i] = isz;
        __icx[i] = icx;
        __icy[i] = icy;
        __icz[i] = icz;
    }
}

