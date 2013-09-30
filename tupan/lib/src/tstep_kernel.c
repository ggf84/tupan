#include "tstep_kernel_common.h"
#include "libtupan.h"


void tstep_kernel(
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
    const REAL eta,
    REAL * restrict _idt_a,
    REAL * restrict _idt_b)
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
        REAL iw2_a = 0;
        REAL iw2_b = 0;
        for (UINT j = 0; j < nj; ++j) {
            REAL jm = _jm[j];
            REAL jrx = _jrx[j];
            REAL jry = _jry[j];
            REAL jrz = _jrz[j];
            REAL je2 = _je2[j];
            REAL jvx = _jvx[j];
            REAL jvy = _jvy[j];
            REAL jvz = _jvz[j];
            tstep_kernel_core(eta,
                              im, irx, iry, irz, ie2, ivx, ivy, ivz,
                              jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                              &iw2_a, &iw2_b);
        }
        _idt_a[i] = eta / sqrt(1 + iw2_a);
        _idt_b[i] = eta / sqrt(1 + iw2_b);
    }
}

