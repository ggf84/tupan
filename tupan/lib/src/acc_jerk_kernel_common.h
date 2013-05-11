#ifndef __ACC_JERK_KERNEL_COMMON_H__
#define __ACC_JERK_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

inline void
acc_jerk_kernel_core(
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
    REAL *iax, REAL *iay, REAL *iaz,
    REAL *ijx, REAL *ijy, REAL *ijz)
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

    REAL e2 = ie2 + je2;                                             // 1 FLOPs

    REAL inv_r2, inv_r3;
    smoothed_inv_r2r3(r2, e2, &inv_r2, &inv_r3);                     // 5 FLOPs

    inv_r3 *= jm;                                                    // 1 FLOPs
    rv *= 3 * inv_r2;                                                // 2 FLOPs

    *iax -= inv_r3 * rx;                                             // 2 FLOPs
    *iay -= inv_r3 * ry;                                             // 2 FLOPs
    *iaz -= inv_r3 * rz;                                             // 2 FLOPs
    *ijx -= inv_r3 * (vx - rv * rx);                                 // 4 FLOPs
    *ijy -= inv_r3 * (vy - rv * ry);                                 // 4 FLOPs
    *ijz -= inv_r3 * (vz - rv * rz);                                 // 4 FLOPs
}
// Total flop count: 43

#endif  // __ACC_JERK_KERNEL_COMMON_H__
