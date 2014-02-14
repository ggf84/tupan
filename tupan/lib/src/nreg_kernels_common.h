#ifndef __NREG_KERNELS_COMMON_H__
#define __NREG_KERNELS_COMMON_H__

#include "common.h"
#include "smoothing.h"

static inline void nreg_Xkernel_core(
    const REALn dt,
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
    REALn *idrx,
    REALn *idry,
    REALn *idrz,
    REALn *iax,
    REALn *iay,
    REALn *iaz,
    REALn *iu)
{
    REALn rx = irx - jrx;                                                       // 1 FLOPs
    REALn ry = iry - jry;                                                       // 1 FLOPs
    REALn rz = irz - jrz;                                                       // 1 FLOPs
    REALn e2 = ie2 + je2;                                                       // 1 FLOPs
    REALn vx = ivx - jvx;                                                       // 1 FLOPs
    REALn vy = ivy - jvy;                                                       // 1 FLOPs
    REALn vz = ivz - jvz;                                                       // 1 FLOPs

    rx += vx * dt;                                                              // 2 FLOPs
    ry += vy * dt;                                                              // 2 FLOPs
    rz += vz * dt;                                                              // 2 FLOPs

    REALn r2 = rx * rx + ry * ry + rz * rz;                                     // 5 FLOPs
    INTn mask = (r2 > 0);

    REALn inv_r1, inv_r3;
    smoothed_inv_r1r3(r2, e2, mask, &inv_r1, &inv_r3);                          // 4 FLOPs

    inv_r3 *= jm;                                                               // 1 FLOPs

    *idrx += jm * rx;                                                           // 2 FLOPs
    *idry += jm * ry;                                                           // 2 FLOPs
    *idrz += jm * rz;                                                           // 2 FLOPs
    *iax -= inv_r3 * rx;                                                        // 2 FLOPs
    *iay -= inv_r3 * ry;                                                        // 2 FLOPs
    *iaz -= inv_r3 * rz;                                                        // 2 FLOPs
    *iu += jm * inv_r1;                                                         // 2 FLOPs
}
// Total flop count: 37

static inline void nreg_Vkernel_core(
    const REALn dt,
    const REALn im,
    const REALn ivx,
    const REALn ivy,
    const REALn ivz,
    const REALn iax,
    const REALn iay,
    const REALn iaz,
    const REALn jm,
    const REALn jvx,
    const REALn jvy,
    const REALn jvz,
    const REALn jax,
    const REALn jay,
    const REALn jaz,
    REALn *idvx,
    REALn *idvy,
    REALn *idvz,
    REALn *ik)
{
    REALn vx = ivx - jvx;                                                       // 1 FLOPs
    REALn vy = ivy - jvy;                                                       // 1 FLOPs
    REALn vz = ivz - jvz;                                                       // 1 FLOPs
    REALn ax = iax - jax;                                                       // 1 FLOPs
    REALn ay = iay - jay;                                                       // 1 FLOPs
    REALn az = iaz - jaz;                                                       // 1 FLOPs

    vx += ax * dt;                                                              // 2 FLOPs
    vy += ay * dt;                                                              // 2 FLOPs
    vz += az * dt;                                                              // 2 FLOPs

    REALn v2 = vx * vx + vy * vy + vz * vz;                                     // 5 FLOPs

    *idvx += jm * vx;                                                           // 2 FLOPs
    *idvy += jm * vy;                                                           // 2 FLOPs
    *idvz += jm * vz;                                                           // 2 FLOPs
    *ik += jm * v2;                                                             // 2 FLOPs
}
// Total flop count: 25

#endif  // __NREG_KERNELS_COMMON_H__
