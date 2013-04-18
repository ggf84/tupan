#ifndef TSTEP_KERNEL_COMMON_H
#define TSTEP_KERNEL_COMMON_H

#include "common.h"
#include "smoothing.h"


inline void
tstep_kernel_core(
    const unsigned int i,
    const unsigned int j,
    const REAL eta,
    const REAL4 irm, const REAL4 ive,
    const REAL4 jrm, const REAL4 jve,
    REAL *iw2,
    REAL *ijw2max,
    int *iid,
    int *jid
    )
{
    REAL4 r;
    r.x = irm.x - jrm.x;                                             // 1 FLOPs
    r.y = irm.y - jrm.y;                                             // 1 FLOPs
    r.z = irm.z - jrm.z;                                             // 1 FLOPs
    r.w = irm.w + jrm.w;                                             // 1 FLOPs
    REAL4 v;
    v.x = ive.x - jve.x;                                             // 1 FLOPs
    v.y = ive.y - jve.y;                                             // 1 FLOPs
    v.z = ive.z - jve.z;                                             // 1 FLOPs
    v.w = ive.w + jve.w;                                             // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL rv = r.x * v.x + r.y * v.y + r.z * v.z;                     // 5 FLOPs
    REAL v2 = v.x * v.x + v.y * v.y + v.z * v.z;                     // 5 FLOPs

    REAL2 ret = smoothed_inv_r2r3(r2, v.w);                          // 5 FLOPs
    REAL inv_r2 = ret.x;
    REAL inv_r3 = ret.y;

    REAL b = 1;
    REAL c = 2;
    REAL d = 1 / (b + c);                                            // 3 FLOPs
    REAL e = (b + c / 2);                                            // 2 FLOPs

    REAL f1 = v2 * inv_r2;                                           // 1 FLOPs
    REAL f2 = r.w * inv_r3;                                          // 1 FLOPs
    REAL w2 = d * (b * f1 + c * f2);                                 // 5 FLOPs
    REAL gamma = 1 + d * (e * f2) / w2;                              // 5 FLOPs
    REAL dln_w = -gamma * rv * inv_r2;                               // 2 FLOPs
    w2 = sqrt(w2);                                                   // 1 FLOPs
    w2 += eta * dln_w;   // factor 1/2 included in 'eta'             // 2 FLOPs
    w2 *= w2;                                                        // 1 FLOPs

    w2 = (r2 > 0) ? (w2):(1);   // It should be (1), not (0).

    int mask = (w2 > *ijw2max);

    *iw2 += w2;                                                      // 1 FLOPs
    *iid = mask ? (i):(*iid);
    *jid = mask ? (j):(*jid);
    *ijw2max = mask ? (w2):(*ijw2max);
}
// Total flop count: 52

#endif  // !TSTEP_KERNEL_COMMON_H
