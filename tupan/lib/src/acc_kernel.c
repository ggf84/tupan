#include "acc_kernel_common.h"


inline void
p2p_acc_kernel(const unsigned int ni,
               const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
               const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
               const unsigned int nj,
               const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
               const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
               REAL *iax, REAL *iay, REAL *iaz)
{
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 irm = {irx[i], iry[i], irz[i], imass[i]};
        REAL4 ive = {ivx[i], ivy[i], ivz[i], ieps2[i]};
        REAL3 ia = (REAL3){0, 0, 0};
        for (j = 0; j < nj; ++j) {
            REAL4 jrm = {jrx[j], jry[j], jrz[j], jmass[j]};
            REAL4 jve = {jvx[j], jvy[j], jvz[j], jeps2[j]};
            ia = p2p_acc_kernel_core(ia, irm, ive, jrm, jve);
        }
        iax[i] = ia.x;
        iay[i] = ia.y;
        iaz[i] = ia.z;
    }
}

