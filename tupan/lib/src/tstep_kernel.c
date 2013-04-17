#include "tstep_kernel_common.h"


inline void
p2p_tstep_kernel(const unsigned int ni,
                 const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
                 const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
                 const unsigned int nj,
                 const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
                 const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
                 const REAL eta,
                 REAL *idt)
{
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 irm = {irx[i], iry[i], irz[i], imass[i]};
        REAL4 ive = {ivx[i], ivy[i], ivz[i], ieps2[i]};
        REAL iomega = (REAL)0;
        for (j = 0; j < nj; ++j) {
            REAL4 jrm = {jrx[j], jry[j], jrz[j], jmass[j]};
            REAL4 jve = {jvx[j], jvy[j], jvz[j], jeps2[j]};
            iomega = p2p_tstep_kernel_core(iomega, irm, ive, jrm, jve, eta);
        }
//        idt[i] = 2 * eta / iomega;
        idt[i] = 2 * eta / sqrt(iomega);
    }
}

