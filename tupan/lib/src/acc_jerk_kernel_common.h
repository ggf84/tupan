#ifndef __ACC_JERK_KERNEL_COMMON_H__
#define __ACC_JERK_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


static inline void
acc_jerk_kernel_core(
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
    REALn *iax,
    REALn *iay,
    REALn *iaz,
    REALn *ijx,
    REALn *ijy,
    REALn *ijz)
{
    REALn rx = irx - jrx;                                                       // 1 FLOPs
    REALn ry = iry - jry;                                                       // 1 FLOPs
    REALn rz = irz - jrz;                                                       // 1 FLOPs
    REALn e2 = ie2 + je2;                                                       // 1 FLOPs
    REALn vx = ivx - jvx;                                                       // 1 FLOPs
    REALn vy = ivy - jvy;                                                       // 1 FLOPs
    REALn vz = ivz - jvz;                                                       // 1 FLOPs
    REALn r2 = rx * rx + ry * ry + rz * rz;                                     // 5 FLOPs
    REALn rv = rx * vx + ry * vy + rz * vz;                                     // 5 FLOPs
    INTn mask = (r2 > 0);

    REALn inv_r2;
    REALn m_r3 = smoothed_m_r3_inv_r2(jm, r2, e2, mask, &inv_r2);               // 5 FLOPs

    REALn alpha = 3 * rv * inv_r2;                                              // 2 FLOPs

    vx -= alpha * rx;                                                           // 2 FLOPs
    vy -= alpha * ry;                                                           // 2 FLOPs
    vz -= alpha * rz;                                                           // 2 FLOPs

    *iax -= m_r3 * rx;                                                          // 2 FLOPs
    *iay -= m_r3 * ry;                                                          // 2 FLOPs
    *iaz -= m_r3 * rz;                                                          // 2 FLOPs
    *ijx -= m_r3 * vx;                                                          // 2 FLOPs
    *ijy -= m_r3 * vy;                                                          // 2 FLOPs
    *ijz -= m_r3 * vz;                                                          // 2 FLOPs
}
// Total flop count: 42


#endif  // __ACC_JERK_KERNEL_COMMON_H__
