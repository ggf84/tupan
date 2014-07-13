#include "tstep_kernel_common.h"


void tstep_kernel(
    const UINT ni,
    const REAL * restrict __im,
    const REAL * restrict __irx,
    const REAL * restrict __iry,
    const REAL * restrict __irz,
    const REAL * restrict __ie2,
    const REAL * restrict __ivx,
    const REAL * restrict __ivy,
    const REAL * restrict __ivz,
    const UINT nj,
    const REAL * restrict __jm,
    const REAL * restrict __jrx,
    const REAL * restrict __jry,
    const REAL * restrict __jrz,
    const REAL * restrict __je2,
    const REAL * restrict __jvx,
    const REAL * restrict __jvy,
    const REAL * restrict __jvz,
    const REAL eta,
    REAL * restrict __idt_a,
    REAL * restrict __idt_b)
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
        REAL iw2_a = 0;
        REAL iw2_b = 0;

        for (UINT j = 0; j < nj; ++j) {
            REAL jm = __jm[j];
            REAL jrx = __jrx[j];
            REAL jry = __jry[j];
            REAL jrz = __jrz[j];
            REAL je2 = __je2[j];
            REAL jvx = __jvx[j];
            REAL jvy = __jvy[j];
            REAL jvz = __jvz[j];
            tstep_kernel_core(
                eta,
                im, irx, iry, irz, ie2, ivx, ivy, ivz,
                jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                &iw2_a, &iw2_b);
        }

        __idt_a[i] = eta / sqrt(fmax((REALn)(1), iw2_a));
        __idt_b[i] = eta / sqrt(fmax((REALn)(1), iw2_b));
    }
}

