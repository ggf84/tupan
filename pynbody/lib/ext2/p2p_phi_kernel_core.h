#ifndef P2P_PHI_KERNEL_CORE_H
#define P2P_PHI_KERNEL_CORE_H

#include"common.h"
#include"smoothing.h"

inline REAL
p2p_phi_kernel_core(REAL phi, REAL4 bi, REAL hi2, REAL4 bj, REAL hj2)
{
    REAL4 r;
    r.x = bi.x - bj.x;                                               // 1 FLOPs
    r.y = bi.y - bj.y;                                               // 1 FLOPs
    r.z = bi.z - bj.z;                                               // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL rinv = phi_smooth(r2, hi2, hj2);                            // 4 FLOPs
    phi -= bj.w * rinv;                                              // 2 FLOPs
    return phi;
}
// Total flop count: 14

#endif  // P2P_PHI_KERNEL_CORE_H

