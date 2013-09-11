#include "acc_kernel_common.h"


inline void acc_kernel(
    const unsigned int ni,
    const REAL *_im,
    const REAL *_irx,
    const REAL *_iry,
    const REAL *_irz,
    const REAL *_ie2,
    const unsigned int nj,
    const REAL *_jm,
    const REAL *_jrx,
    const REAL *_jry,
    const REAL *_jrz,
    const REAL *_je2,
    REAL *_iax,
    REAL *_iay,
    REAL *_iaz)
{
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL im = _im[i];
        REAL irx = _irx[i];
        REAL iry = _iry[i];
        REAL irz = _irz[i];
        REAL ie2 = _ie2[i];
        REAL iax = 0;
        REAL iay = 0;
        REAL iaz = 0;
        for (j = 0; j < nj; ++j) {
            REAL jm = _jm[j];
            REAL jrx = _jrx[j];
            REAL jry = _jry[j];
            REAL jrz = _jrz[j];
            REAL je2 = _je2[j];
            acc_kernel_core(im, irx, iry, irz, ie2,
                            jm, jrx, jry, jrz, je2,
                            &iax, &iay, &iaz);
        }
        _iax[i] = iax;
        _iay[i] = iay;
        _iaz[i] = iaz;
    }
}

