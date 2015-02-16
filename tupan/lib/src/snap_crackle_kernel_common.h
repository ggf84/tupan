#ifndef __SNAP_CRACKLE_KERNEL_COMMON_H__
#define __SNAP_CRACKLE_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

#define _3_2 (3/(real_t)(2))


static inline void
snap_crackle_kernel_core(
    real_tn const im,
    real_tn const irx,
    real_tn const iry,
    real_tn const irz,
    real_tn const ie2,
    real_tn const ivx,
    real_tn const ivy,
    real_tn const ivz,
    real_tn const iax,
    real_tn const iay,
    real_tn const iaz,
    real_tn const ijx,
    real_tn const ijy,
    real_tn const ijz,
    real_tn const jm,
    real_tn const jrx,
    real_tn const jry,
    real_tn const jrz,
    real_tn const je2,
    real_tn const jvx,
    real_tn const jvy,
    real_tn const jvz,
    real_tn const jax,
    real_tn const jay,
    real_tn const jaz,
    real_tn const jjx,
    real_tn const jjy,
    real_tn const jjz,
    real_tn *isx,
    real_tn *isy,
    real_tn *isz,
    real_tn *icx,
    real_tn *icy,
    real_tn *icz)
{
    real_tn rx = irx - jrx;                                                     // 1 FLOPs
    real_tn ry = iry - jry;                                                     // 1 FLOPs
    real_tn rz = irz - jrz;                                                     // 1 FLOPs
    real_tn e2 = ie2 + je2;                                                     // 1 FLOPs
    real_tn vx = ivx - jvx;                                                     // 1 FLOPs
    real_tn vy = ivy - jvy;                                                     // 1 FLOPs
    real_tn vz = ivz - jvz;                                                     // 1 FLOPs
    real_tn ax = iax - jax;                                                     // 1 FLOPs
    real_tn ay = iay - jay;                                                     // 1 FLOPs
    real_tn az = iaz - jaz;                                                     // 1 FLOPs
    real_tn jx = ijx - jjx;                                                     // 1 FLOPs
    real_tn jy = ijy - jjy;                                                     // 1 FLOPs
    real_tn jz = ijz - jjz;                                                     // 1 FLOPs
    real_tn r2 = rx * rx + ry * ry + rz * rz;                                   // 5 FLOPs
    real_tn rv = rx * vx + ry * vy + rz * vz;                                   // 5 FLOPs
    real_tn v2 = vx * vx + vy * vy + vz * vz;                                   // 5 FLOPs
    real_tn rj = rx * jx + ry * jy + rz * jz;                                   // 5 FLOPs
    real_tn ra = rx * ax + ry * ay + rz * az;                                   // 5 FLOPs
    real_tn va = vx * ax + vy * ay + vz * az;                                   // 5 FLOPs
    int_tn mask = (r2 > 0);

    real_tn inv_r2;
    real_tn m_r3 = smoothed_m_r3_inv_r2(jm, r2, e2, mask, &inv_r2);             // 5 FLOPs

    real_tn alpha = rv * inv_r2;                                                // 1 FLOPs
    real_tn alpha2 = alpha * alpha;                                             // 1 FLOPs
    real_tn beta = 3 * ((v2 + ra) * inv_r2 + alpha2);                           // 4 FLOPs
    real_tn gamma = (3 * va + rj) * inv_r2 + alpha * (beta - 4 * alpha2);       // 7 FLOPs

    alpha *= 3;                                                                 // 1 FLOPs
    gamma *= 3;                                                                 // 1 FLOPs

    vx -= alpha * rx;                                                           // 2 FLOPs
    vy -= alpha * ry;                                                           // 2 FLOPs
    vz -= alpha * rz;                                                           // 2 FLOPs

    alpha *= 2;                                                                 // 1 FLOPs
    ax -= alpha * vx;                                                           // 2 FLOPs
    ay -= alpha * vy;                                                           // 2 FLOPs
    az -= alpha * vz;                                                           // 2 FLOPs
    ax -= beta * rx;                                                            // 2 FLOPs
    ay -= beta * ry;                                                            // 2 FLOPs
    az -= beta * rz;                                                            // 2 FLOPs

    alpha *= _3_2;                                                              // 1 FLOPs
    beta *= 3;                                                                  // 1 FLOPs
    jx -= alpha * ax;                                                           // 2 FLOPs
    jy -= alpha * ay;                                                           // 2 FLOPs
    jz -= alpha * az;                                                           // 2 FLOPs
    jx -= beta * vx;                                                            // 2 FLOPs
    jy -= beta * vy;                                                            // 2 FLOPs
    jz -= beta * vz;                                                            // 2 FLOPs
    jx -= gamma * rx;                                                           // 2 FLOPs
    jy -= gamma * ry;                                                           // 2 FLOPs
    jz -= gamma * rz;                                                           // 2 FLOPs

    *isx -= m_r3 * ax;                                                          // 2 FLOPs
    *isy -= m_r3 * ay;                                                          // 2 FLOPs
    *isz -= m_r3 * az;                                                          // 2 FLOPs
    *icx -= m_r3 * jx;                                                          // 2 FLOPs
    *icy -= m_r3 * jy;                                                          // 2 FLOPs
    *icz -= m_r3 * jz;                                                          // 2 FLOPs
}
// Total flop count: 114


#endif  // __SNAP_CRACKLE_KERNEL_COMMON_H__
