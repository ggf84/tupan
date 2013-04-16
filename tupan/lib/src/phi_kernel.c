#include"common.h"
#include"smoothing.h"


inline REAL
p2p_phi_kernel_core(REAL iphi,
                    const REAL4 irm, const REAL4 ive,
                    const REAL4 jrm, const REAL4 jve)
{
    REAL3 r;
    r.x = irm.x - jrm.x;                                             // 1 FLOPs
    r.y = irm.y - jrm.y;                                             // 1 FLOPs
    r.z = irm.z - jrm.z;                                             // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL inv_r = smoothed_inv_r1(r2, ive.w + jve.w);                 // 4+1 FLOPs
    iphi -= jrm.w * inv_r;                                           // 2 FLOPs
    return iphi;
}
// Total flop count: 15


inline void
p2p_phi_kernel(const unsigned int ni,
               const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
               const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
               const unsigned int nj,
               const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
               const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
               REAL *iphi)
{
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 irm = {irx[i], iry[i], irz[i], imass[i]};
        REAL4 ive = {ivx[i], ivy[i], ivz[i], ieps2[i]};
        REAL ip = (REAL)0;
        for (j = 0; j < nj; ++j) {
            REAL4 jrm = {jrx[j], jry[j], jrz[j], jmass[j]};
            REAL4 jve = {jvx[j], jvy[j], jvz[j], jeps2[j]};
            ip = p2p_phi_kernel_core(ip, irm, ive, jrm, jve);
        }
        iphi[i] = ip;
    }
}

