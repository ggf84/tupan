#ifndef __SNAP_CRACKLE_KERNEL_COMMON_H__
#define __SNAP_CRACKLE_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

static inline void snap_crackle_kernel_core(
    const REALn im,
    const REALn irx,
    const REALn iry,
    const REALn irz,
    const REALn ie2,
    const REALn ivx,
    const REALn ivy,
    const REALn ivz,
    const REALn iAx,
    const REALn iAy,
    const REALn iAz,
    const REALn iJx,
    const REALn iJy,
    const REALn iJz,
    const REALn jm,
    const REALn jrx,
    const REALn jry,
    const REALn jrz,
    const REALn je2,
    const REALn jvx,
    const REALn jvy,
    const REALn jvz,
    const REALn jAx,
    const REALn jAy,
    const REALn jAz,
    const REALn jJx,
    const REALn jJy,
    const REALn jJz,
    REALn *iSx, REALn *iSy, REALn *iSz,
    REALn *iCx, REALn *iCy, REALn *iCz)
{
    REALn rx = irx - jrx;                                                       // 1 FLOPs
    REALn ry = iry - jry;                                                       // 1 FLOPs
    REALn rz = irz - jrz;                                                       // 1 FLOPs
    REALn e2 = ie2 + je2;                                                       // 1 FLOPs
    REALn vx = ivx - jvx;                                                       // 1 FLOPs
    REALn vy = ivy - jvy;                                                       // 1 FLOPs
    REALn vz = ivz - jvz;                                                       // 1 FLOPs
    REALn Ax = iAx - jAx;                                                       // 1 FLOPs
    REALn Ay = iAy - jAy;                                                       // 1 FLOPs
    REALn Az = iAz - jAz;                                                       // 1 FLOPs
    REALn Jx = iJx - jJx;                                                       // 1 FLOPs
    REALn Jy = iJy - jJy;                                                       // 1 FLOPs
    REALn Jz = iJz - jJz;                                                       // 1 FLOPs
    REALn r2 = rx * rx + ry * ry + rz * rz;                                     // 5 FLOPs
    REALn rv = rx * vx + ry * vy + rz * vz;                                     // 5 FLOPs
    REALn v2 = vx * vx + vy * vy + vz * vz;                                     // 5 FLOPs
    REALn rJ = rx * Jx + ry * Jy + rz * Jz;                                     // 5 FLOPs
    REALn rA = rx * Ax + ry * Ay + rz * Az;                                     // 5 FLOPs
    REALn vA = vx * Ax + vy * Ay + vz * Az;                                     // 5 FLOPs

    REALn inv_r2, inv_r3;
    smoothed_inv_r2r3(r2, e2, &inv_r2, &inv_r3);                                // 4 FLOPs

    REALn alpha = rv * inv_r2;                                                  // 1 FLOPs
    REALn alpha2 = alpha * alpha;                                               // 1 FLOPs
    REALn beta = 3 * ((v2 + rA) * inv_r2 + alpha2);                             // 4 FLOPs
    REALn gamma = (3 * vA + rJ) * inv_r2 + alpha * (beta - 4 * alpha2);         // 7 FLOPs

    alpha *= 3;                                                                 // 1 FLOPs
    gamma *= 3;                                                                 // 1 FLOPs

    vx -= alpha * rx;                                                           // 2 FLOPs
    vy -= alpha * ry;                                                           // 2 FLOPs
    vz -= alpha * rz;                                                           // 2 FLOPs

    REALn sw1 = 2 * alpha;                                                      // 1 FLOPs
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
