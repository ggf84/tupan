#ifndef __ACC_JERK_KERNEL_COMMON_H__
#define __ACC_JERK_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

inline void acc_jerk_kernel_core(
    const REALn im,
    const REALn irx,
    const REALn iry,
    const REALn irz,
    const REALn ie2,
    const REALn ivx,
    const REALn ivy,
    const REALn ivz,
    const REALn jm,
    const REALn jrx,
    const REALn jry,
    const REALn jrz,
    const REALn je2,
    const REALn jvx,
    const REALn jvy,
    const REALn jvz,
    REALn *iAx, REALn *iAy, REALn *iAz,
    REALn *iJx, REALn *iJy, REALn *iJz)
{
    REALn rx, ry, rz, e2;
    rx = irx - jrx;                                                             // 1 FLOPs
    ry = iry - jry;                                                             // 1 FLOPs
    rz = irz - jrz;                                                             // 1 FLOPs
    e2 = ie2 + je2;                                                             // 1 FLOPs
    REALn vx, vy, vz;
    vx = ivx - jvx;                                                             // 1 FLOPs
    vy = ivy - jvy;                                                             // 1 FLOPs
    vz = ivz - jvz;                                                             // 1 FLOPs
    REALn r2 = rx * rx + ry * ry + rz * rz;                                     // 5 FLOPs
    REALn rv = rx * vx + ry * vy + rz * vz;                                     // 5 FLOPs

    REALn inv_r2, inv_r3;
    smoothed_inv_r2r3(r2, e2, &inv_r2, &inv_r3);                                // 4 FLOPs

    REALn alpha = 3 * rv * inv_r2;                                              // 2 FLOPs

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
