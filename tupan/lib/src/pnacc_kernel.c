#include "pnacc_kernel_common.h"
#include "libtupan.h"


void pnacc_kernel(
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
    UINT order,
    const REAL inv1,
    const REAL inv2,
    const REAL inv3,
    const REAL inv4,
    const REAL inv5,
    const REAL inv6,
    const REAL inv7,
    REAL * restrict _ipnax,
    REAL * restrict _ipnay,
    REAL * restrict _ipnaz)
{
    CLIGHT clight = CLIGHT_Init(order, inv1, inv2, inv3, inv4, inv5, inv6, inv7);
    for (UINT i = 0; i < ni; ++i) {
        REAL im = _im[i];
        REAL irx = _irx[i];
        REAL iry = _iry[i];
        REAL irz = _irz[i];
        REAL ie2 = _ie2[i];
        REAL ivx = _ivx[i];
        REAL ivy = _ivy[i];
        REAL ivz = _ivz[i];
        REAL ipnax = 0;
        REAL ipnay = 0;
        REAL ipnaz = 0;
        for (UINT j = 0; j < nj; ++j) {
            pnacc_kernel_core(im, irx, iry, irz, ie2, ivx, ivy, ivz,
                              _jm[j], _jrx[j], _jry[j], _jrz[j],
                              _je2[j], _jvx[j], _jvy[j], _jvz[j],
                              clight,
                              &ipnax, &ipnay, &ipnaz);
        }
        _ipnax[i] = ipnax;
        _ipnay[i] = ipnay;
        _ipnaz[i] = ipnaz;
    }
}

