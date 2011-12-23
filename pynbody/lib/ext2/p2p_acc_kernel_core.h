#ifndef P2P_ACC_KERNEL_CORE_H
#define P2P_ACC_KERNEL_CORE_H

#include"common.h"
#include"smoothing.h"


inline REAL4
p2p_acc_kernel_core(REAL4 acc,
                    const REAL4 bi, const REAL4 vi,
                    const REAL4 bj, const REAL4 vj,
                    const REAL eta)
{
    REAL4 r;
    r.x = bi.x - bj.x;                                               // 1 FLOPs
    r.y = bi.y - bj.y;                                               // 1 FLOPs
    r.z = bi.z - bj.z;                                               // 1 FLOPs
    r.w = bi.w + bj.w;                                               // 1 FLOPs
    REAL4 v;
    v.x = vi.x - vj.x;                                               // 1 FLOPs
    v.y = vi.y - vj.y;                                               // 1 FLOPs
    v.z = vi.z - vj.z;                                               // 1 FLOPs
    v.w = vi.w + vj.w;                                               // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL rv = r.x * v.x + r.y * v.y + r.z * v.z;                     // 5 FLOPs
    REAL rinv3 = acc_smooth(r2, vi.w, vj.w);                         // 6 FLOPs

    REAL omega2 = (r.w * rinv3);                                     // 1 FLOPs

    REAL dln_omega = -1.5 * rv / (r2 + v.w);                         // 3 FLOPs
    REAL symm_factor = (1.0 + 0.5 * eta * dln_omega);                // 3 FLOPs
    omega2 *= (symm_factor * symm_factor);                           // 2 FLOPs

    acc.w += omega2;                                                 // 1 FLOPs

    rinv3 *= bj.w;                                                   // 1 FLOPs
    acc.x -= rinv3 * r.x;                                            // 2 FLOPs
    acc.y -= rinv3 * r.y;                                            // 2 FLOPs
    acc.z -= rinv3 * r.z;                                            // 2 FLOPs
    return acc;
}
// Total flop count: 24

#endif  // P2P_ACC_KERNEL_CORE_H

