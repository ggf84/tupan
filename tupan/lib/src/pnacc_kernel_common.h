#ifndef PNACC_KERNEL_COMMON_H
#define PNACC_KERNEL_COMMON_H

#include "common.h"
#include "smoothing.h"
#include "pn_terms.h"


inline REAL3
pnacc_kernel_core(
    REAL3 ipna,
    const REAL4 irm, const REAL4 ive,
    const REAL4 jrm, const REAL4 jve,
    const CLIGHT clight
    )
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

#endif  // !PNACC_KERNEL_COMMON_H
