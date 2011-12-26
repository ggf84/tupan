#ifndef P2P_ACC_KERNEL_CORE_H
#define P2P_ACC_KERNEL_CORE_H

#include"common.h"
#include"smoothing.h"


inline REAL4
p2p_acc_kernel_core(REAL4 acc,
                    const REAL4 ri, const REAL4 vi,
                    const REAL4 rj, const REAL4 vj,
                    const REAL eta)
{
    REAL4 r;
    r.x = ri.x - rj.x;                                               // 1 FLOPs
    r.y = ri.y - rj.y;                                               // 1 FLOPs
    r.z = ri.z - rj.z;                                               // 1 FLOPs
    r.w = ri.w + rj.w;                                               // 1 FLOPs
    REAL4 v;
    v.x = vi.x - vj.x;                                               // 1 FLOPs
    v.y = vi.y - vj.y;                                               // 1 FLOPs
    v.z = vi.z - vj.z;                                               // 1 FLOPs
    v.w = vi.w + vj.w;                                               // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL rv = r.x * v.x + r.y * v.y + r.z * v.z;                     // 5 FLOPs
    REAL inv_r3 = acc_smooth(r2, v.w);                               // 5 FLOPs

    REAL omega2 = (r.w * inv_r3);                                    // 1 FLOPs

    REAL dln_omega = -1.5 * rv / (r2 + v.w);                         // 3 FLOPs
    dln_omega = (r2 > 0) ? (dln_omega):(0);
    dln_omega = (dln_omega < 0) ? (-dln_omega):(dln_omega);
    REAL symm_factor = 1 + eta * dln_omega;                          // 2 FLOPs
    omega2 *= symm_factor;                                           // 1 FLOPs

    acc.w += omega2;                                                 // 1 FLOPs

    inv_r3 *= rj.w;                                                  // 1 FLOPs
    acc.x -= inv_r3 * r.x;                                           // 2 FLOPs
    acc.y -= inv_r3 * r.y;                                           // 2 FLOPs
    acc.z -= inv_r3 * r.z;                                           // 2 FLOPs
    return acc;
}
// Total flop count: 38

#endif  // P2P_ACC_KERNEL_CORE_H

