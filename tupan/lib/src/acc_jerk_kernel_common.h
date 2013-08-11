#ifndef __ACC_JERK_KERNEL_COMMON_H__
#define __ACC_JERK_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

inline void acc_jerk_kernel_core(
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
    REAL *iAx, REAL *iAy, REAL *iAz,
    REAL *iJx, REAL *iJy, REAL *iJz)
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

    REAL e2 = ie2 + je2;                                                        // 1 FLOPs

    REAL inv_r2, inv_r3;
    smoothed_inv_r2r3(r2, e2, &inv_r2, &inv_r3);                                // 4 FLOPs

    REAL alpha = 3 * rv * inv_r2;                                               // 2 FLOPs

    vx -= alpha * rx;                                                           // 2 FLOPs
    vy -= alpha * ry;                                                           // 2 FLOPs
    vz -= alpha * rz;                                                           // 2 FLOPs

    inv_r3 *= jm;                                                               // 1 FLOPs

    *iAx -= inv_r3 * rx;                                                        // 2 FLOPs
    *iAy -= inv_r3 * ry;                                                        // 2 FLOPs
    *iAz -= inv_r3 * rz;                                                        // 2 FLOPs
    *iJx -= inv_r3 * vx;                                                        // 2 FLOPs
    *iJy -= inv_r3 * vy;                                                        // 2 FLOPs
    *iJz -= inv_r3 * vz;                                                        // 2 FLOPs
}
// Total flop count: 42

#endif  // __ACC_JERK_KERNEL_COMMON_H__
