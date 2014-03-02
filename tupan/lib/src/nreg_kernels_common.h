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

    REALn m_r1;
    REALn m_r3 = smoothed_m_r3_m_r1(jm, r2, e2, mask, &m_r1);                   // 5 FLOPs

    *idrx += jm * rx;                                                           // 2 FLOPs
    *idry += jm * ry;                                                           // 2 FLOPs
    *idrz += jm * rz;                                                           // 2 FLOPs
    *iax -= m_r3 * rx;                                                          // 2 FLOPs
    *iay -= m_r3 * ry;                                                          // 2 FLOPs
    *iaz -= m_r3 * rz;                                                          // 2 FLOPs
    *iu += m_r1;                                                                // 1 FLOPs
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
