#ifndef __TSTEP_KERNEL_COMMON_H__
#define __TSTEP_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

inline void tstep_kernel_core(
    const REAL eta,
    const REAL im,
    const REAL irx,
    const REAL iry,
    const REAL irz,
    const REAL ie2,
    const REAL ivx,
    const REAL ivy,
    const REAL ivz,
    const REAL jm,
    const REAL jrx,
    const REAL jry,
    const REAL jrz,
    const REAL je2,
    const REAL jvx,
    const REAL jvy,
    const REAL jvz,
    REAL *iw2_a,
    REAL *iw2_b)
{
    REAL rx, ry, rz;
    rx = irx - jrx;                                                             // 1 FLOPs
    ry = iry - jry;                                                             // 1 FLOPs
    rz = irz - jrz;                                                             // 1 FLOPs
    REAL vx, vy, vz;
    vx = ivx - jvx;                                                             // 1 FLOPs
    vy = ivy - jvy;                                                             // 1 FLOPs
    vz = ivz - jvz;                                                             // 1 FLOPs
    REAL r2 = rx * rx + ry * ry + rz * rz;                                      // 5 FLOPs
    REAL rv = rx * vx + ry * vy + rz * vz;                                      // 5 FLOPs
    REAL v2 = vx * vx + vy * vy + vz * vz;                                      // 5 FLOPs

    REAL m = im + jm;                                                           // 1 FLOPs
    REAL e2 = ie2 + je2;                                                        // 1 FLOPs

    REAL inv_r1, inv_r2;
    smoothed_inv_r1r2(r2, e2, &inv_r1, &inv_r2);                                // 3 FLOPs

    REAL a = inv_r2;
    REAL b = 2 * inv_r2;                                                        // 1 FLOPs
    REAL c = (a + b / 2);                                                       // 2 FLOPs

    REAL phi = m * inv_r1;                                                      // 1 FLOPs
    REAL w2 = a * v2 + b * phi;                                                 // 3 FLOPs
    REAL gamma = w2 + c * phi;                                                  // 2 FLOPs
    REAL dln_w = gamma * rv * inv_r2;                                           // 2 FLOPs
    w2 -= (eta / sqrt(w2)) * dln_w;                                             // 4 FLOPs

    w2 = select(((REAL)0), w2, (INT)(r2 > 0));

    *iw2_a += w2;                                                               // 1 FLOPs
    *iw2_b = max(w2, *iw2_b);
}
// Total flop count: 42

#endif  // __TSTEP_KERNEL_COMMON_H__
