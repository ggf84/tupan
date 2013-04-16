#include"common.h"
#include"smoothing.h"


inline REAL3
p2p_acc_kernel_core(REAL3 ia,
                    const REAL4 irm, const REAL4 ive,
                    const REAL4 jrm, const REAL4 jve)
{
    REAL3 r;
    r.x = irm.x - jrm.x;                                             // 1 FLOPs
    r.y = irm.y - jrm.y;                                             // 1 FLOPs
    r.z = irm.z - jrm.z;                                             // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL inv_r3 = smoothed_inv_r3(r2, ive.w + jve.w);                // 5+1 FLOPs

    inv_r3 *= jrm.w;                                                 // 1 FLOPs

    ia.x -= inv_r3 * r.x;                                            // 2 FLOPs
    ia.y -= inv_r3 * r.y;                                            // 2 FLOPs
    ia.z -= inv_r3 * r.z;                                            // 2 FLOPs
    return ia;
}
// Total flop count: 21


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

