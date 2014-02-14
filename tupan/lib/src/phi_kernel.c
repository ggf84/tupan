#include "phi_kernel_common.h"
#include "libtupan.h"


void phi_kernel(
    const UINT ni,
    const REAL * restrict _im,
    const REAL * restrict _irx,
    const REAL * restrict _iry,
    const REAL * restrict _irz,
    const REAL * restrict _ie2,
    const UINT nj,
    const REAL * restrict _jm,
    const REAL * restrict _jrx,
    const REAL * restrict _jry,
    const REAL * restrict _jrz,
    const REAL * restrict _je2,
    REAL * restrict _iphi)
{
    for (UINT i = 0; i < ni; ++i) {
        REAL im = _im[i];
        REAL irx = _irx[i];
        REAL iry = _iry[i];
        REAL irz = _irz[i];
        REAL ie2 = _ie2[i];
        REAL iphi = 0;
        for (UINT j = 0; j < nj; ++j) {
            phi_kernel_core(im, irx, iry, irz, ie2,
                            _jm[j], _jrx[j], _jry[j], _jrz[j], _je2[j],
                            &iphi);
        }
        _iphi[i] = iphi;
    }
}

