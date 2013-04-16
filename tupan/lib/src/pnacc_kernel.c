#include"common.h"
#include"smoothing.h"
#include"pn_terms.h"


inline REAL3
p2p_pnacc_kernel_core(REAL3 ipna,
                      const REAL4 irm, const REAL4 ive,
                      const REAL4 jrm, const REAL4 jve,
                      const CLIGHT clight)
{
    REAL3 r;
    r.x = irm.x - jrm.x;                                             // 1 FLOPs
    r.y = irm.y - jrm.y;                                             // 1 FLOPs
    r.z = irm.z - jrm.z;                                             // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs

    REAL3 v;
    v.x = ive.x - jve.x;                                             // 1 FLOPs
    v.y = ive.y - jve.y;                                             // 1 FLOPs
    v.z = ive.z - jve.z;                                             // 1 FLOPs
    REAL v2 = v.x * v.x + v.y * v.y + v.z * v.z;                     // 5 FLOPs

    REAL3 ret = smoothed_inv_r1r2r3(r2, ive.w + jve.w);              // 5+1 FLOPs
    REAL inv_r = ret.x;
    REAL inv_r2 = ret.y;
    REAL inv_r3 = ret.z;

    REAL mij = irm.w + jrm.w;                                        // 1 FLOPs
    REAL r_sch = 2 * mij * clight.inv2;
    REAL gamma = r_sch * inv_r;

    if (16777216*gamma > 1) {
//    if (mij > 1.9) {
//        printf("mi: %e, mj: %e, mij: %e\n", rmi.w, rmj.w, mij);
        REAL3 vi = {ive.x, ive.y, ive.z};
        REAL3 vj = {jve.x, jve.y, jve.z};
        REAL2 pn = p2p_pnterms(irm.w, jrm.w,
                               r, v, v2,
                               vi, vj,
                               inv_r, inv_r2, inv_r3,
                               clight);                              // ? FLOPs

        ipna.x += pn.x * r.x + pn.y * v.x;                           // 4 FLOPs
        ipna.y += pn.x * r.y + pn.y * v.y;                           // 4 FLOPs
        ipna.z += pn.x * r.z + pn.y * v.z;                           // 4 FLOPs
    }

    return ipna;
}
// Total flop count: 36+?+???


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

