#include "pnacc_kernel_common.h"


inline void pnacc_kernel(
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
    unsigned int order,
    const REAL inv1,
    const REAL inv2,
    const REAL inv3,
    const REAL inv4,
    const REAL inv5,
    const REAL inv6,
    const REAL inv7,
    REAL *_ipnax,
    REAL *_ipnay,
    REAL *_ipnaz)
{
    CLIGHT clight = (CLIGHT){.order=order, .inv1=inv1,
                             .inv2=inv2, .inv3=inv3,
                             .inv4=inv4, .inv5=inv5,
                             .inv6=inv6, .inv7=inv7};
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
        REAL ipnax = 0;
        REAL ipnay = 0;
        REAL ipnaz = 0;
        for (j = 0; j < nj; ++j) {
            REAL jm = _jm[j];
            REAL jrx = _jrx[j];
            REAL jry = _jry[j];
            REAL jrz = _jrz[j];
            REAL je2 = _je2[j];
            REAL jvx = _jvx[j];
            REAL jvy = _jvy[j];
            REAL jvz = _jvz[j];
            pnacc_kernel_core(im, irx, iry, irz, ie2, ivx, ivy, ivz,
                              jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                              clight,
                              &ipnax, &ipnay, &ipnaz);
        }
        _ipnax[i] = ipnax;
        _ipnay[i] = ipnay;
        _ipnaz[i] = ipnaz;
    }
}

