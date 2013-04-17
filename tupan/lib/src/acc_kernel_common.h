#ifndef ACC_KERNEL_COMMON_H
#define ACC_KERNEL_COMMON_H

#include "common.h"
#include "smoothing.h"


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

#endif  // !ACC_KERNEL_COMMON_H
