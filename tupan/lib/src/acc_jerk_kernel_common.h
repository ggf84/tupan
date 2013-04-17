#ifndef ACC_JERK_KERNEL_COMMON_H
#define ACC_JERK_KERNEL_COMMON_H

#include "common.h"
#include "smoothing.h"


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

#endif  // !ACC_JERK_KERNEL_COMMON_H
