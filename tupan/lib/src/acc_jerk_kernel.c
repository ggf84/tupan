#include "acc_jerk_kernel_common.h"
#include "libtupan.h"


void acc_jerk_kernel(
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
    REAL * restrict _iax,
    REAL * restrict _iay,
    REAL * restrict _iaz,
    REAL * restrict _ijx,
    REAL * restrict _ijy,
    REAL * restrict _ijz)
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
        REAL iax = 0;
        REAL iay = 0;
        REAL iaz = 0;
        REAL ijx = 0;
        REAL ijy = 0;
        REAL ijz = 0;
        for (UINT j = 0; j < nj; ++j) {
            acc_jerk_kernel_core(im, irx, iry, irz, ie2, ivx, ivy, ivz,
                                 _jm[j], _jrx[j], _jry[j], _jrz[j],
                                 _je2[j], _jvx[j], _jvy[j], _jvz[j],
                                 &iax, &iay, &iaz,
                                 &ijx, &ijy, &ijz);
        }
        _iax[i] = iax;
        _iay[i] = iay;
        _iaz[i] = iaz;
        _ijx[i] = ijx;
        _ijy[i] = ijy;
        _ijz[i] = ijz;
    }
}

