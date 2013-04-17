#ifndef PHI_KERNEL_COMMON_H
#define PHI_KERNEL_COMMON_H

#include "common.h"
#include "smoothing.h"


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

#endif  // !PHI_KERNEL_COMMON_H
