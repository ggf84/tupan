#include "sakura_kernel_common.h"


inline void
sakura_kernel(
    const unsigned int ni,
    const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
    const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
    const unsigned int nj,
    const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
    const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
    const REAL dt,
    REAL *idrx, REAL *idry, REAL *idrz,
    REAL *idvx, REAL *idvy, REAL *idvz
    )
{
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 irm = {irx[i], iry[i], irz[i], imass[i]};
        REAL4 ive = {ivx[i], ivy[i], ivz[i], ieps2[i]};
        REAL8 idrdv = (REAL8){0, 0, 0, 0, 0, 0, 0, 0};
        for (j = 0; j < nj; ++j) {
            REAL4 jrm = {jrx[j], jry[j], jrz[j], jmass[j]};
            REAL4 jve = {jvx[j], jvy[j], jvz[j], jeps2[j]};
            idrdv = sakura_kernel_core(idrdv, irm, ive, jrm, jve, dt);
        }
        idrx[i] = idrdv.s0;
        idry[i] = idrdv.s1;
        idrz[i] = idrdv.s2;
        idvx[i] = idrdv.s4;
        idvy[i] = idrdv.s5;
        idvz[i] = idrdv.s6;
    }
}

