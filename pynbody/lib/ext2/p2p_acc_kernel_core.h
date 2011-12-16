#ifndef P2P_ACC_KERNEL_CORE_H
#define P2P_ACC_KERNEL_CORE_H

#include"common.h"
#include"smoothing.h"

inline REAL4
p2p_acc_kernel_core(REAL4 acc, REAL4 bi, REAL hi2, REAL4 bj, REAL hj2)
{
    REAL4 r;
    r.x = bi.x - bj.x;                                               // 1 FLOPs
    r.y = bi.y - bj.y;                                               // 1 FLOPs
    r.z = bi.z - bj.z;                                               // 1 FLOPs
    r.w = bi.w + bj.w;                                               // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL rinv3 = acc_smooth(r2, hi2, hj2);                           // 6 FLOPs

//    acc.w += r.w * rinv3;                                            // 2 FLOPs
    acc.w += bi.w * bj.w * phi_smooth(r2, hi2, hj2);                 // 2 FLOPs
//    acc.w += bi.w * bj.w * (r2+hi2+hj2) * rinv3;                     // 2 FLOPs

    rinv3 *= bj.w;                                                   // 1 FLOPs
    acc.x -= rinv3 * r.x;                                            // 2 FLOPs
    acc.y -= rinv3 * r.y;                                            // 2 FLOPs
    acc.z -= rinv3 * r.z;                                            // 2 FLOPs
    return acc;
}
// Total flop count: 24

#endif  // P2P_ACC_KERNEL_CORE_H

