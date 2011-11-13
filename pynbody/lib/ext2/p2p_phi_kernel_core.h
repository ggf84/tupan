#ifndef P2P_PHI_KERNEL_CORE_H
#define P2P_PHI_KERNEL_CORE_H

#include"common.h"


inline REAL
p2p_phi_kernel_core(REAL phi, REAL4 bi, REAL ieps2, REAL4 bj, REAL jeps2)
{
    REAL4 dr;
    dr.x = bi.x - bj.x;                                              // 1 FLOPs
    dr.y = bi.y - bj.y;                                              // 1 FLOPs
    dr.z = bi.z - bj.z;                                              // 1 FLOPs
    REAL dr2 = dr.z * dr.z + (dr.y * dr.y + dr.x * dr.x);            // 5 FLOPs
    REAL eps2 = ieps2 + jeps2;                                       // 1 FLOPs
    REAL rinv = rsqrt(dr2 + eps2);                                   // 3 FLOPs
    phi -= bj.w * ((dr2 > 0) ? rinv:0);                              // 2 FLOPs
    return phi;
}
// Total flop count: 14

#endif  // P2P_PHI_KERNEL_CORE_H

