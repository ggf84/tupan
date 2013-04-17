#include "pnacc_kernel_common.h"


inline void
p2p_pnacc_kernel(const unsigned int ni,
                 const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
                 const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
                 const unsigned int nj,
                 const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
                 const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
                 unsigned int order, const REAL inv1,
                 const REAL inv2, const REAL inv3,
                 const REAL inv4, const REAL inv5,
                 const REAL inv6, const REAL inv7,
                 REAL *ipnax, REAL *ipnay, REAL *ipnaz)
{
    CLIGHT clight = {.order=order, .inv1=inv1,
                     .inv2=inv2, .inv3=inv3,
                     .inv4=inv4, .inv5=inv5,
                     .inv6=inv6, .inv7=inv7};
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 irm = {irx[i], iry[i], irz[i], imass[i]};
        REAL4 ive = {ivx[i], ivy[i], ivz[i], ieps2[i]};
        REAL3 ipna = (REAL3){0, 0, 0};
        for (j = 0; j < nj; ++j) {
            REAL4 jrm = {jrx[j], jry[j], jrz[j], jmass[j]};
            REAL4 jve = {jvx[j], jvy[j], jvz[j], jeps2[j]};
            ipna = p2p_pnacc_kernel_core(ipna, irm, ive, jrm, jve, clight);
        }
        ipnax[i] = ipna.x;
        ipnay[i] = ipna.y;
        ipnaz[i] = ipna.z;
    }
}

