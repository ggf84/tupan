#ifndef P2P_ACC_KERNEL_CORE_H
#define P2P_ACC_KERNEL_CORE_H

#include"common.h"


inline REAL4
p2p_acc_kernel_core(REAL4 acc, REAL4 bi, REAL ieps2, REAL4 bj, REAL jeps2)
{
    REAL4 dr;
    dr.x = bi.x - bj.x;                                              // 1 FLOPs
    dr.y = bi.y - bj.y;                                              // 1 FLOPs
    dr.z = bi.z - bj.z;                                              // 1 FLOPs
    dr.w = bi.w + bj.w;                                              // 1 FLOPs
    REAL dr2 = dr.z * dr.z + (dr.y * dr.y + dr.x * dr.x);            // 5 FLOPs
    REAL eps2 = ieps2 + jeps2;                                       // 1 FLOPs
    REAL rinv = rsqrt(dr2 + eps2);                                   // 3 FLOPs
    REAL r3inv = rinv = ((dr2 > 0) ? rinv:0);
    r3inv *= rinv * rinv;                                            // 2 FLOPs
    acc.w += dr.w * r3inv;                                           // 2 FLOPs
    r3inv *= bj.w;                                                   // 1 FLOPs
    acc.x -= r3inv * dr.x;                                           // 2 FLOPs
    acc.y -= r3inv * dr.y;                                           // 2 FLOPs
    acc.z -= r3inv * dr.z;                                           // 2 FLOPs
    return acc;
}
// Total flop count: 24

#endif  // P2P_ACC_KERNEL_CORE_H

