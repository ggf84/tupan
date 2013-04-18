#include "tstep_kernel_common.h"


inline void
tstep_kernel(
    const unsigned int ni,
    const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
    const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
    const unsigned int nj,
    const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
    const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
    const REAL eta,
    REAL *idt,
    REAL *ijdtmin
    )
{
    unsigned int i, j;
    int iid = -1;
    int jid = -1;
    REAL ijw2max = (REAL)0;
    for (i = 0; i < ni; ++i) {
        REAL4 irm = {irx[i], iry[i], irz[i], imass[i]};
        REAL4 ive = {ivx[i], ivy[i], ivz[i], ieps2[i]};
        REAL iw2 = (REAL)0;
        for (j = 0; j < nj; ++j) {
            REAL4 jrm = {jrx[j], jry[j], jrz[j], jmass[j]};
            REAL4 jve = {jvx[j], jvy[j], jvz[j], jeps2[j]};
            tstep_kernel_core(i, j, eta, irm, ive, jrm, jve, &iw2, &ijw2max, &iid, &jid);
        }
        idt[i] = 2 * eta / sqrt(iw2);
    }
    ijdtmin[0] = iid;
    ijdtmin[1] = jid;
    ijdtmin[2] = 2 * eta / sqrt(ijw2max);
}

