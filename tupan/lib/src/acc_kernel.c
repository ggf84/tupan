#include "acc_kernel_common.h"


inline void
acc_kernel(
    const unsigned int ni,
    const REAL *_im,
    const REAL *_irx,
    const REAL *_iry,
    const REAL *_irz,
    const REAL *_ie2,
    const REAL *_ivx,
    const REAL *_ivy,
    const REAL *_ivz,
    const unsigned int nj,
    const REAL *_jm,
    const REAL *_jrx,
    const REAL *_jry,
    const REAL *_jrz,
    const REAL *_je2,
    const REAL *_jvx,
    const REAL *_jvy,
    const REAL *_jvz,
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
        REAL ivx = _ivx[i];
        REAL ivy = _ivy[i];
        REAL ivz = _ivz[i];
        REAL iax = 0;
        REAL iay = 0;
        REAL iaz = 0;
        for (j = 0; j < nj; ++j) {
            REAL jm = _jm[j];
            REAL jrx = _jrx[j];
            REAL jry = _jry[j];
            REAL jrz = _jrz[j];
            REAL je2 = _je2[j];
            REAL jvx = _jvx[j];
            REAL jvy = _jvy[j];
            REAL jvz = _jvz[j];
            acc_kernel_core(im, irx, iry, irz, ie2, ivx, ivy, ivz,
                            jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                            &iax, &iay, &iaz);
        }
        _iax[i] = iax;
        _iay[i] = iay;
        _iaz[i] = iaz;
    }
}

