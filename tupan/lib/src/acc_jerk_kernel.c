#include"common.h"
#include"smoothing.h"


inline REAL8
p2p_acc_jerk_kernel_core(REAL8 iaj,
                         const REAL4 irm, const REAL4 ive,
                         const REAL4 jrm, const REAL4 jve)
{
    REAL3 r;
    r.x = irm.x - jrm.x;                                             // 1 FLOPs
    r.y = irm.y - jrm.y;                                             // 1 FLOPs
    r.z = irm.z - jrm.z;                                             // 1 FLOPs
    REAL4 v;
    v.x = ive.x - jve.x;                                             // 1 FLOPs
    v.y = ive.y - jve.y;                                             // 1 FLOPs
    v.z = ive.z - jve.z;                                             // 1 FLOPs
    v.w = ive.w + jve.w;                                             // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL rv = r.x * v.x + r.y * v.y + r.z * v.z;                     // 5 FLOPs
    REAL2 ret = smoothed_inv_r2r3(r2, v.w);                          // 5 FLOPs
    REAL inv_r2 = ret.x;
    REAL inv_r3 = ret.y;

    inv_r3 *= jrm.w;                                                 // 1 FLOPs
    rv *= 3 * inv_r2;                                                // 2 FLOPs

    iaj.s0 -= inv_r3 * r.x;                                          // 2 FLOPs
    iaj.s1 -= inv_r3 * r.y;                                          // 2 FLOPs
    iaj.s2 -= inv_r3 * r.z;                                          // 2 FLOPs
    iaj.s3  = 0;
    iaj.s4 -= inv_r3 * (v.x - rv * r.x);                             // 4 FLOPs
    iaj.s5 -= inv_r3 * (v.y - rv * r.y);                             // 4 FLOPs
    iaj.s6 -= inv_r3 * (v.z - rv * r.z);                             // 4 FLOPs
    iaj.s7  = 0;
    return iaj;
}
// Total flop count: 43


inline void
p2p_acc_jerk_kernel(const unsigned int ni,
                    const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
                    const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
                    const unsigned int nj,
                    const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
                    const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
                    REAL *iax, REAL *iay, REAL *iaz,
                    REAL *ijx, REAL *ijy, REAL *ijz)
{
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 irm = {irx[i], iry[i], irz[i], imass[i]};
        REAL4 ive = {ivx[i], ivy[i], ivz[i], ieps2[i]};
        REAL8 iaj = (REAL8){0, 0, 0, 0, 0, 0, 0, 0};
        for (j = 0; j < nj; ++j) {
            REAL4 jrm = {jrx[j], jry[j], jrz[j], jmass[j]};
            REAL4 jve = {jvx[j], jvy[j], jvz[j], jeps2[j]};
            iaj = p2p_acc_jerk_kernel_core(iaj, irm, ive, jrm, jve);
        }
        iax[i] = iaj.s0;
        iay[i] = iaj.s1;
        iaz[i] = iaj.s2;
        ijx[i] = iaj.s4;
        ijy[i] = iaj.s5;
        ijz[i] = iaj.s6;
    }
}

