#include "snap_crackle_kernel_common.h"


void
snap_crackle_kernel(
    uint_t const ni,
    real_t const __im[restrict],
    real_t const __irx[restrict],
    real_t const __iry[restrict],
    real_t const __irz[restrict],
    real_t const __ie2[restrict],
    real_t const __ivx[restrict],
    real_t const __ivy[restrict],
    real_t const __ivz[restrict],
    real_t const __iax[restrict],
    real_t const __iay[restrict],
    real_t const __iaz[restrict],
    real_t const __ijx[restrict],
    real_t const __ijy[restrict],
    real_t const __ijz[restrict],
    uint_t const nj,
    real_t const __jm[restrict],
    real_t const __jrx[restrict],
    real_t const __jry[restrict],
    real_t const __jrz[restrict],
    real_t const __je2[restrict],
    real_t const __jvx[restrict],
    real_t const __jvy[restrict],
    real_t const __jvz[restrict],
    real_t const __jax[restrict],
    real_t const __jay[restrict],
    real_t const __jaz[restrict],
    real_t const __jjx[restrict],
    real_t const __jjy[restrict],
    real_t const __jjz[restrict],
    real_t __isx[restrict],
    real_t __isy[restrict],
    real_t __isz[restrict],
    real_t __icx[restrict],
    real_t __icy[restrict],
    real_t __icz[restrict])
{
    for (uint_t i = 0; i < ni; ++i) {
        real_t im = __im[i];
        real_t irx = __irx[i];
        real_t iry = __iry[i];
        real_t irz = __irz[i];
        real_t ie2 = __ie2[i];
        real_t ivx = __ivx[i];
        real_t ivy = __ivy[i];
        real_t ivz = __ivz[i];
        real_t iax = __iax[i];
        real_t iay = __iay[i];
        real_t iaz = __iaz[i];
        real_t ijx = __ijx[i];
        real_t ijy = __ijy[i];
        real_t ijz = __ijz[i];
        real_t isx = 0;
        real_t isy = 0;
        real_t isz = 0;
        real_t icx = 0;
        real_t icy = 0;
        real_t icz = 0;

        for (uint_t j = 0; j < nj; ++j) {
            real_t jm = __jm[j];
            real_t jrx = __jrx[j];
            real_t jry = __jry[j];
            real_t jrz = __jrz[j];
            real_t je2 = __je2[j];
            real_t jvx = __jvx[j];
            real_t jvy = __jvy[j];
            real_t jvz = __jvz[j];
            real_t jax = __jax[j];
            real_t jay = __jay[j];
            real_t jaz = __jaz[j];
            real_t jjx = __jjx[j];
            real_t jjy = __jjy[j];
            real_t jjz = __jjz[j];
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

