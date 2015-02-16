#ifndef __ACC_JERK_KERNEL_COMMON_H__
#define __ACC_JERK_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


static inline void
acc_jerk_kernel_core(
    real_tn const im,
    real_tn const irx,
    real_tn const iry,
    real_tn const irz,
    real_tn const ie2,
    real_tn const ivx,
    real_tn const ivy,
    real_tn const ivz,
    real_tn const jm,
    real_tn const jrx,
    real_tn const jry,
    real_tn const jrz,
    real_tn const je2,
    real_tn const jvx,
    real_tn const jvy,
    real_tn const jvz,
    real_tn *iax,
    real_tn *iay,
    real_tn *iaz,
    real_tn *ijx,
    real_tn *ijy,
    real_tn *ijz)
{
    real_tn rx = irx - jrx;                                                     // 1 FLOPs
    real_tn ry = iry - jry;                                                     // 1 FLOPs
    real_tn rz = irz - jrz;                                                     // 1 FLOPs
    real_tn e2 = ie2 + je2;                                                     // 1 FLOPs
    real_tn vx = ivx - jvx;                                                     // 1 FLOPs
    real_tn vy = ivy - jvy;                                                     // 1 FLOPs
    real_tn vz = ivz - jvz;                                                     // 1 FLOPs
    real_tn r2 = rx * rx + ry * ry + rz * rz;                                   // 5 FLOPs
    real_tn rv = rx * vx + ry * vy + rz * vz;                                   // 5 FLOPs
    int_tn mask = (r2 > 0);

    real_tn inv_r2;
    real_tn m_r3 = smoothed_m_r3_inv_r2(jm, r2, e2, mask, &inv_r2);             // 5 FLOPs

    real_tn alpha = 3 * rv * inv_r2;                                            // 2 FLOPs

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
