#ifndef __TSTEP_KERNEL_COMMON_H__
#define __TSTEP_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

inline void
tstep_kernel_core(
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
    rx = irx - jrx;                                                  // 1 FLOPs
    ry = iry - jry;                                                  // 1 FLOPs
    rz = irz - jrz;                                                  // 1 FLOPs
    REAL vx, vy, vz;
    vx = ivx - jvx;                                                  // 1 FLOPs
    vy = ivy - jvy;                                                  // 1 FLOPs
    vz = ivz - jvz;                                                  // 1 FLOPs
    REAL r2 = rx * rx + ry * ry + rz * rz;                           // 5 FLOPs
    REAL rv = rx * vx + ry * vy + rz * vz;                           // 5 FLOPs
    REAL v2 = vx * vx + vy * vy + vz * vz;                           // 5 FLOPs

    REAL inv_r2, inv_r3;
    smoothed_inv_r2r3(r2, ie2 + je2, &inv_r2, &inv_r3);              // 5+1 FLOPs

    REAL b = 1;
    REAL c = 2;
    REAL d = 1 / (b + c);                                            // 3 FLOPs
    REAL e = (b + c / 2);                                            // 2 FLOPs

    REAL m = im + jm;                                                // 1 FLOPs
    REAL f1 = v2 * inv_r2;                                           // 1 FLOPs
    REAL f2 = m * inv_r3;                                            // 1 FLOPs
    REAL w2 = d * (b * f1 + c * f2);                                 // 5 FLOPs
    REAL gamma = 1 + d * (e * f2) / w2;                              // 5 FLOPs
    REAL dln_w = -gamma * rv * inv_r2;                               // 2 FLOPs
    w2 = sqrt(w2);                                                   // 1 FLOPs
    w2 += eta * dln_w;          // factor 1/2 included in 'eta'      // 2 FLOPs
    w2 *= w2;                                                        // 1 FLOPs

    w2 = (r2 > 0) ? (w2):(1);   // It should be (1), not (0).

    *iw2_a += w2;                                                    // 1 FLOPs
    *iw2_b = (w2 > *iw2_b) ? (w2):(*iw2_b);
}
// Total flop count: 52

#endif  // __TSTEP_KERNEL_COMMON_H__
