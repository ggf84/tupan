#ifndef GRAVITY_KERNELS_H
#define GRAVITY_KERNELS_H

#include"common.h"
#include"smoothing.h"
#include"pn_terms.h"


//
// p2p_phi_kernel_core
////////////////////////////////////////////////////////////////////////////////
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


//
// p2p_acc_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL3
p2p_acc_kernel_core(REAL3 acc,
                    const REAL4 ri, const REAL hi2,
                    const REAL4 rj, const REAL hj2)
{
    REAL4 r;
    r.x = ri.x - rj.x;                                               // 1 FLOPs
    r.y = ri.y - rj.y;                                               // 1 FLOPs
    r.z = ri.z - rj.z;                                               // 1 FLOPs
    r.w = hi2 + hj2;                                                 // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL inv_r2 = 1 / (r2 + r.w);                                    // 2 FLOPs
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);                                       // 1 FLOPs
    REAL inv_r3 = inv_r * inv_r2;                                    // 1 FLOPs

    inv_r3 *= rj.w;                                                  // 1 FLOPs

    acc.x -= inv_r3 * r.x;                                           // 2 FLOPs
    acc.y -= inv_r3 * r.y;                                           // 2 FLOPs
    acc.z -= inv_r3 * r.z;                                           // 2 FLOPs
    return acc;
}
// Total flop count: 20


//
// p2p_acctstep_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL4
p2p_acctstep_kernel_core(REAL4 acctstep,
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
    REAL v2 = v.x * v.x + v.y * v.y + v.z * v.z;                     // 5 FLOPs
    REAL rv = r.x * v.x + r.y * v.y + r.z * v.z;                     // 5 FLOPs
    REAL inv_r2 = 1 / (r2 + v.w);                                    // 2 FLOPs
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);                                       // 1 FLOPs
    REAL inv_r3 = inv_r * inv_r2;                                    // 1 FLOPs

    REAL omega2_a = v2 * inv_r2;                                     // 1 FLOPs
    REAL omega2_b = r.w * inv_r3;                                    // 1 FLOPs
    REAL omega2 = omega2_a + omega2_b;                               // 1 FLOPs
    REAL inv_32omega2 = 3 / (2 * omega2);                            // 2 FLOPs
    inv_32omega2 = (r2 > 0) ? (inv_32omega2):(0);
    REAL weighting = (1 + omega2_b * inv_32omega2);                  // 2 FLOPs
    REAL dln_omega = -weighting * rv * inv_r2;                       // 2 FLOPs
    REAL omega = sqrt(omega2);                                       // 1 FLOPs
    omega += eta * dln_omega;   // factor 1/2 included in 'eta'      // 2 FLOPs

    inv_r3 *= rj.w;                                                  // 1 FLOPs

    acctstep.x -= inv_r3 * r.x;                                      // 2 FLOPs
    acctstep.y -= inv_r3 * r.y;                                      // 2 FLOPs
    acctstep.z -= inv_r3 * r.z;                                      // 2 FLOPs
    acctstep.w = (omega > acctstep.w) ? (omega):(acctstep.w);
    return acctstep;
}
// Total flop count: 46


//
// p2p_tstep_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL
p2p_tstep_kernel_core(REAL inv_tstep,
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
    REAL v2 = v.x * v.x + v.y * v.y + v.z * v.z;                     // 5 FLOPs
    REAL rv = r.x * v.x + r.y * v.y + r.z * v.z;                     // 5 FLOPs
    REAL inv_r2 = 1 / (r2 + v.w);                                    // 2 FLOPs
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);                                       // 1 FLOPs
    REAL inv_r3 = inv_r * inv_r2;                                    // 1 FLOPs

    REAL omega2_a = v2 * inv_r2;                                     // 1 FLOPs
    REAL omega2_b = r.w * inv_r3;                                    // 1 FLOPs
    REAL omega2 = omega2_a + omega2_b;                               // 1 FLOPs
    REAL inv_32omega2 = 3 / (2 * omega2);                            // 2 FLOPs
    inv_32omega2 = (r2 > 0) ? (inv_32omega2):(0);
    REAL weighting = (1 + omega2_b * inv_32omega2);                  // 2 FLOPs
    REAL dln_omega = -weighting * rv * inv_r2;                       // 2 FLOPs
    REAL omega = sqrt(omega2);                                       // 1 FLOPs
    omega += eta * dln_omega;   // factor 1/2 included in 'eta'      // 2 FLOPs

    inv_tstep = (omega > inv_tstep) ? (omega):(inv_tstep);
    return inv_tstep;
}
// Total flop count: 39


//
// p2p_pnacc_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL3
p2p_pnacc_kernel_core(REAL3 pnacc,
                      const REAL4 ri, const REAL4 vi,
                      const REAL4 rj, const REAL4 vj,
                      const CLIGHT clight)
{
    REAL3 r;
    r.x = ri.x - rj.x;                                               // 1 FLOPs
    r.y = ri.y - rj.y;                                               // 1 FLOPs
    r.z = ri.z - rj.z;                                               // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs

    REAL mi = ri.w;
    REAL mj = rj.w;

    REAL3 v;
    v.x = vi.x - vj.x;                                               // 1 FLOPs
    v.y = vi.y - vj.y;                                               // 1 FLOPs
    v.z = vi.z - vj.z;                                               // 1 FLOPs
    REAL v2 = v.x * v.x + v.y * v.y + v.z * v.z;                     // 5 FLOPs

    REAL vi2 = vi.w;
    REAL vj2 = vj.w;

    REAL3 vixyz = {vi.x, vi.y, vi.z};
    REAL3 vjxyz = {vj.x, vj.y, vj.z};

    REAL inv_r2 = 1 / r2;                                            // 1 FLOPs
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);                                       // 1 FLOPs

    REAL3 n;
    n.x = r.x * inv_r;                                               // 1 FLOPs
    n.y = r.y * inv_r;                                               // 1 FLOPs
    n.z = r.z * inv_r;                                               // 1 FLOPs

    REAL2 pn = p2p_pnterms(mi, mj,
                           inv_r, inv_r2,
                           n, v, v2,
                           vi2, vixyz,
                           vj2, vjxyz,
                           clight);                                  // ? FLOPs

    pnacc.x += pn.x * n.x + pn.y * v.x;                              // 4 FLOPs
    pnacc.y += pn.x * n.y + pn.y * v.y;                              // 4 FLOPs
    pnacc.z += pn.x * n.z + pn.y * v.z;                              // 4 FLOPs

    return pnacc;
}
// Total flop count: 36+???


#endif  // GRAVITY_KERNELS_H

