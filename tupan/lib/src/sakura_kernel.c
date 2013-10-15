#include "sakura_kernel_common.h"
#include "libtupan.h"


void sakura_kernel(
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
    const INT flag,
    REAL * restrict _idrx,
    REAL * restrict _idry,
    REAL * restrict _idrz,
    REAL * restrict _idvx,
    REAL * restrict _idvy,
    REAL * restrict _idvz)
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
        REAL idvx = 0;
        REAL idvy = 0;
        REAL idvz = 0;
        for (UINT j = 0; j < nj; ++j) {
            sakura_kernel_core(dt, flag,
                               im, irx, iry, irz, ie2, ivx, ivy, ivz,
                               _jm[j], _jrx[j], _jry[j], _jrz[j],
                               _je2[j], _jvx[j], _jvy[j], _jvz[j],
                               &idrx, &idry, &idrz,
                               &idvx, &idvy, &idvz);
        }
        _idrx[i] = idrx;
        _idry[i] = idry;
        _idrz[i] = idrz;
        _idvx[i] = idvx;
        _idvy[i] = idvy;
        _idvz[i] = idvz;
    }
}

