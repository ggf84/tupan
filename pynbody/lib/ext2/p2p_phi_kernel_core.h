#ifndef P2P_PHI_KERNEL_CORE_H
#define P2P_PHI_KERNEL_CORE_H

#include"common.h"
#include"smoothing.h"


inline REAL
p2p_phi_kernel_core(REAL phi,
                    const REAL4 ri, const REAL hi2,
                    const REAL4 rj, const REAL hj2)
{
    REAL4 r;
    r.x = ri.x - rj.x;                                               // 1 FLOPs
    r.y = ri.y - rj.y;                                               // 1 FLOPs
    r.z = ri.z - rj.z;                                               // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL inv_r = phi_smooth(r2, hi2 + hj2);                          // 4 FLOPs
    phi -= rj.w * inv_r;                                             // 2 FLOPs
    return phi;
}
// Total flop count: 14

#endif  // P2P_PHI_KERNEL_CORE_H

