#include "snap_crackle_kernel_common.h"


void
snap_crackle_kernel(
    UINT const ni,
    REAL const __im[restrict],
    REAL const __irx[restrict],
    REAL const __iry[restrict],
    REAL const __irz[restrict],
    REAL const __ie2[restrict],
    REAL const __ivx[restrict],
    REAL const __ivy[restrict],
    REAL const __ivz[restrict],
    REAL const __iax[restrict],
    REAL const __iay[restrict],
    REAL const __iaz[restrict],
    REAL const __ijx[restrict],
    REAL const __ijy[restrict],
    REAL const __ijz[restrict],
    UINT const nj,
    REAL const __jm[restrict],
    REAL const __jrx[restrict],
    REAL const __jry[restrict],
    REAL const __jrz[restrict],
    REAL const __je2[restrict],
    REAL const __jvx[restrict],
    REAL const __jvy[restrict],
    REAL const __jvz[restrict],
    REAL const __jax[restrict],
    REAL const __jay[restrict],
    REAL const __jaz[restrict],
    REAL const __jjx[restrict],
    REAL const __jjy[restrict],
    REAL const __jjz[restrict],
    REAL __isx[restrict],
    REAL __isy[restrict],
    REAL __isz[restrict],
    REAL __icx[restrict],
    REAL __icy[restrict],
    REAL __icz[restrict])
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
            REAL jm = __jm[j];
            REAL jrx = __jrx[j];
            REAL jry = __jry[j];
            REAL jrz = __jrz[j];
            REAL je2 = __je2[j];
            REAL jvx = __jvx[j];
            REAL jvy = __jvy[j];
            REAL jvz = __jvz[j];
            REAL jax = __jax[j];
            REAL jay = __jay[j];
            REAL jaz = __jaz[j];
            REAL jjx = __jjx[j];
            REAL jjy = __jjy[j];
            REAL jjz = __jjz[j];
            snap_crackle_kernel_core(
                im, irx, iry, irz, ie2, ivx, ivy, ivz,
                iax, iay, iaz, ijx, ijy, ijz,
                jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                jax, jay, jaz, jjx, jjy, jjz,
                &isx, &isy, &isz, &icx, &icy, &icz);
        }

        __isx[i] = isx;
        __isy[i] = isy;
        __isz[i] = isz;
        __icx[i] = icx;
        __icy[i] = icy;
        __icz[i] = icz;
    }
}

