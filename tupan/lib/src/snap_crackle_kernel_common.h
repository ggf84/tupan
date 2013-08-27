#ifndef __SNAP_CRACKLE_KERNEL_COMMON_H__
#define __SNAP_CRACKLE_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

inline void snap_crackle_kernel_core(
    const REAL im,
    const REAL irx,
    const REAL iry,
    const REAL irz,
    const REAL ie2,
    const REAL ivx,
    const REAL ivy,
    const REAL ivz,
    const REAL iAx,
    const REAL iAy,
    const REAL iAz,
    const REAL iJx,
    const REAL iJy,
    const REAL iJz,
    const REAL jm,
    const REAL jrx,
    const REAL jry,
    const REAL jrz,
    const REAL je2,
    const REAL jvx,
    const REAL jvy,
    const REAL jvz,
    const REAL jAx,
    const REAL jAy,
    const REAL jAz,
    const REAL jJx,
    const REAL jJy,
    const REAL jJz,
    REAL *iSx, REAL *iSy, REAL *iSz,
    REAL *iCx, REAL *iCy, REAL *iCz)
{
    REAL rx, ry, rz;
    rx = irx - jrx;                                                             // 1 FLOPs
    ry = iry - jry;                                                             // 1 FLOPs
    rz = irz - jrz;                                                             // 1 FLOPs
    REAL vx, vy, vz;
    vx = ivx - jvx;                                                             // 1 FLOPs
    vy = ivy - jvy;                                                             // 1 FLOPs
    vz = ivz - jvz;                                                             // 1 FLOPs
    REAL Ax, Ay, Az;
    Ax = iAx - jAx;                                                             // 1 FLOPs
    Ay = iAy - jAy;                                                             // 1 FLOPs
    Az = iAz - jAz;                                                             // 1 FLOPs
    REAL Jx, Jy, Jz;
    Jx = iJx - jJx;                                                             // 1 FLOPs
    Jy = iJy - jJy;                                                             // 1 FLOPs
    Jz = iJz - jJz;                                                             // 1 FLOPs
    REAL r2 = rx * rx + ry * ry + rz * rz;                                      // 5 FLOPs
    REAL v2 = vx * vx + vy * vy + vz * vz;                                      // 5 FLOPs
    REAL rv = rx * vx + ry * vy + rz * vz;                                      // 5 FLOPs
    REAL rA = rx * Ax + ry * Ay + rz * Az;                                      // 5 FLOPs
    REAL rJ = rx * Jx + ry * Jy + rz * Jz;                                      // 5 FLOPs
    REAL vA = vx * Ax + vy * Ay + vz * Az;                                      // 5 FLOPs

    REAL e2 = ie2 + je2;                                                        // 1 FLOPs

    REAL inv_r2, inv_r3;
    smoothed_inv_r2r3(r2, e2, &inv_r2, &inv_r3);                                // 4 FLOPs

    REAL alpha = rv * inv_r2;                                                   // 1 FLOPs
    REAL alpha2 = alpha * alpha;                                                // 1 FLOPs
    REAL beta = (v2 + rA) * inv_r2 + alpha2;                                    // 3 FLOPs
    beta *= 3;                                                                  // 1 FLOPs
    REAL gamma = (3 * vA + rJ) * inv_r2 + alpha * (beta - 4 * alpha2);          // 7 FLOPs

    alpha *= 3;                                                                 // 1 FLOPs
    gamma *= 3;                                                                 // 1 FLOPs

    vx -= alpha * rx;                                                           // 2 FLOPs
    vy -= alpha * ry;                                                           // 2 FLOPs
    vz -= alpha * rz;                                                           // 2 FLOPs

    REAL sw1 = 2 * alpha;                                                       // 1 FLOPs
    Ax -= sw1 * vx + beta * rx;                                                 // 4 FLOPs
    Ay -= sw1 * vy + beta * ry;                                                 // 4 FLOPs
    Az -= sw1 * vz + beta * rz;                                                 // 4 FLOPs

    alpha *= 3;                                                                 // 1 FLOPs
    beta *= 3;                                                                  // 1 FLOPs
    Jx -= alpha * Ax + beta * vx + gamma * rx;                                  // 6 FLOPs
    Jy -= alpha * Ay + beta * vy + gamma * ry;                                  // 6 FLOPs
    Jz -= alpha * Az + beta * vz + gamma * rz;                                  // 6 FLOPs

    inv_r3 *= jm;                                                               // 1 FLOPs

    *iSx -= inv_r3 * Ax;                                                        // 2 FLOPs
    *iSy -= inv_r3 * Ay;                                                        // 2 FLOPs
    *iSz -= inv_r3 * Az;                                                        // 2 FLOPs
    *iCx -= inv_r3 * Jx;                                                        // 2 FLOPs
    *iCy -= inv_r3 * Jy;                                                        // 2 FLOPs
    *iCz -= inv_r3 * Jz;                                                        // 2 FLOPs
}
// Total flop count: 114

#endif  // __SNAP_CRACKLE_KERNEL_COMMON_H__
