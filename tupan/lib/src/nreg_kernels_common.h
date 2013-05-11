#ifndef __NREG_KERNELS_COMMON_H__
#define __NREG_KERNELS_COMMON_H__

#include "common.h"
#include "smoothing.h"

inline void
nreg_Xkernel_core(
    const REAL dt,
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
    REAL *idrx,
    REAL *idry,
    REAL *idrz,
    REAL *iax,
    REAL *iay,
    REAL *iaz,
    REAL *iu)
{
    REAL rx, ry, rz;
    rx = irx - jrx;                                                  // 1 FLOPs
    ry = iry - jry;                                                  // 1 FLOPs
    rz = irz - jrz;                                                  // 1 FLOPs
    REAL vx, vy, vz;
    vx = ivx - jvx;                                                  // 1 FLOPs
    vy = ivy - jvy;                                                  // 1 FLOPs
    vz = ivz - jvz;                                                  // 1 FLOPs

    REAL mij = im * jm;                                              // 1 FLOPs

    rx += vx * dt;                                                   // 2 FLOPs
    ry += vy * dt;                                                   // 2 FLOPs
    rz += vz * dt;                                                   // 2 FLOPs

    REAL r2 = rx * rx + ry * ry + rz * rz;                           // 5 FLOPs

    REAL e2 = ie2 + je2;                                             // 1 FLOPs

    REAL inv_r1, inv_r3;
    smoothed_inv_r1r3(r2, e2, &inv_r1, &inv_r3);                     // 5 FLOPs

    rx *= jm;                                                        // 1 FLOPs
    ry *= jm;                                                        // 1 FLOPs
    rz *= jm;                                                        // 1 FLOPs

    *idrx += rx;                                                     // 1 FLOPs
    *idry += ry;                                                     // 1 FLOPs
    *idrz += rz;                                                     // 1 FLOPs
    *iax -= inv_r3 * rx;                                             // 2 FLOPs
    *iay -= inv_r3 * ry;                                             // 2 FLOPs
    *iaz -= inv_r3 * rz;                                             // 2 FLOPs
    *iu += mij * inv_r1;                                             // 2 FLOPs
}
// Total flop count: 38

inline void
nreg_Vkernel_core(
    const REAL dt,
    const REAL im,
    const REAL ivx,
    const REAL ivy,
    const REAL ivz,
    const REAL iax,
    const REAL iay,
    const REAL iaz,
    const REAL jm,
    const REAL jvx,
    const REAL jvy,
    const REAL jvz,
    const REAL jax,
    const REAL jay,
    const REAL jaz,
    REAL *idvx,
    REAL *idvy,
    REAL *idvz,
    REAL *ik)
{
    REAL vx, vy, vz;
    vx = ivx - jvx;                                                  // 1 FLOPs
    vy = ivy - jvy;                                                  // 1 FLOPs
    vz = ivz - jvz;                                                  // 1 FLOPs
    REAL ax, ay, az;
    ax = iax - jax;                                                  // 1 FLOPs
    ay = iay - jay;                                                  // 1 FLOPs
    az = iaz - jaz;                                                  // 1 FLOPs

    REAL mij = im * jm;                                              // 1 FLOPs

    vx += ax * dt;                                                   // 2 FLOPs
    vy += ay * dt;                                                   // 2 FLOPs
    vz += az * dt;                                                   // 2 FLOPs

    REAL v2 = vx * vx + vy * vy + vz * vz;                           // 5 FLOPs

    vx *= jm;                                                        // 1 FLOPs
    vy *= jm;                                                        // 1 FLOPs
    vz *= jm;                                                        // 1 FLOPs

    *idvx += vx;                                                     // 1 FLOPs
    *idvy += vy;                                                     // 1 FLOPs
    *idvz += vz;                                                     // 1 FLOPs
    *ik += mij * v2;                                                 // 2 FLOPs
}
// Total flop count: 26

#endif  // __NREG_KERNELS_COMMON_H__
