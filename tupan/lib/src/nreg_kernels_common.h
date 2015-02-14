#ifndef __NREG_KERNELS_COMMON_H__
#define __NREG_KERNELS_COMMON_H__

#include "common.h"
#include "smoothing.h"


static inline void
nreg_Xkernel_core(
    REALn const dt,
    REALn const im,
    REALn const irx,
    REALn const iry,
    REALn const irz,
    REALn const ie2,
    REALn const ivx,
    REALn const ivy,
    REALn const ivz,
    REALn const jm,
    REALn const jrx,
    REALn const jry,
    REALn const jrz,
    REALn const je2,
    REALn const jvx,
    REALn const jvy,
    REALn const jvz,
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

    REALn m_r3;
    REALn m_r1 = smoothed_m_r1_m_r3(jm, r2, e2, mask, &m_r3);                   // 5 FLOPs

    *idrx += jm * rx;                                                           // 2 FLOPs
    *idry += jm * ry;                                                           // 2 FLOPs
    *idrz += jm * rz;                                                           // 2 FLOPs
    *iax -= m_r3 * rx;                                                          // 2 FLOPs
    *iay -= m_r3 * ry;                                                          // 2 FLOPs
    *iaz -= m_r3 * rz;                                                          // 2 FLOPs
    *iu += m_r1;                                                                // 1 FLOPs
}
// Total flop count: 37


static inline void
nreg_Vkernel_core(
    REALn const dt,
    REALn const im,
    REALn const ivx,
    REALn const ivy,
    REALn const ivz,
    REALn const iax,
    REALn const iay,
    REALn const iaz,
    REALn const jm,
    REALn const jvx,
    REALn const jvy,
    REALn const jvz,
    REALn const jax,
    REALn const jay,
    REALn const jaz,
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
