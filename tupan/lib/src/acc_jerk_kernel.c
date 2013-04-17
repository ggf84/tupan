#include "acc_jerk_kernel_common.h"


inline void
acc_jerk_kernel(
    const unsigned int ni,
    const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
    const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
    const unsigned int nj,
    const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
    const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
    REAL *iax, REAL *iay, REAL *iaz,
    REAL *ijx, REAL *ijy, REAL *ijz
    )
{
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 irm = {irx[i], iry[i], irz[i], imass[i]};
        REAL4 ive = {ivx[i], ivy[i], ivz[i], ieps2[i]};
        REAL8 iaj = (REAL8){0, 0, 0, 0, 0, 0, 0, 0};
        for (j = 0; j < nj; ++j) {
            REAL4 jrm = {jrx[j], jry[j], jrz[j], jmass[j]};
            REAL4 jve = {jvx[j], jvy[j], jvz[j], jeps2[j]};
            iaj = acc_jerk_kernel_core(iaj, irm, ive, jrm, jve);
        }
        iax[i] = iaj.s0;
        iay[i] = iaj.s1;
        iaz[i] = iaj.s2;
        ijx[i] = iaj.s4;
        ijy[i] = iaj.s5;
        ijz[i] = iaj.s6;
    }
}

