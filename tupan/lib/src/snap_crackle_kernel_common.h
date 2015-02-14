#ifndef __SNAP_CRACKLE_KERNEL_COMMON_H__
#define __SNAP_CRACKLE_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

#define _3_2 (3/(REAL)(2))


static inline void
snap_crackle_kernel_core(
    REALn const im,
    REALn const irx,
    REALn const iry,
    REALn const irz,
    REALn const ie2,
    REALn const ivx,
    REALn const ivy,
    REALn const ivz,
    REALn const iax,
    REALn const iay,
    REALn const iaz,
    REALn const ijx,
    REALn const ijy,
    REALn const ijz,
    REALn const jm,
    REALn const jrx,
    REALn const jry,
    REALn const jrz,
    REALn const je2,
    REALn const jvx,
    REALn const jvy,
    REALn const jvz,
    REALn const jax,
    REALn const jay,
    REALn const jaz,
    REALn const jjx,
    REALn const jjy,
    REALn const jjz,
    REALn *isx,
    REALn *isy,
    REALn *isz,
    REALn *icx,
    REALn *icy,
    REALn *icz)
{
    REALn rx = irx - jrx;                                                       // 1 FLOPs
    REALn ry = iry - jry;                                                       // 1 FLOPs
    REALn rz = irz - jrz;                                                       // 1 FLOPs
    REALn e2 = ie2 + je2;                                                       // 1 FLOPs
    REALn vx = ivx - jvx;                                                       // 1 FLOPs
    REALn vy = ivy - jvy;                                                       // 1 FLOPs
    REALn vz = ivz - jvz;                                                       // 1 FLOPs
    REALn ax = iax - jax;                                                       // 1 FLOPs
    REALn ay = iay - jay;                                                       // 1 FLOPs
    REALn az = iaz - jaz;                                                       // 1 FLOPs
    REALn jx = ijx - jjx;                                                       // 1 FLOPs
    REALn jy = ijy - jjy;                                                       // 1 FLOPs
    REALn jz = ijz - jjz;                                                       // 1 FLOPs
    REALn r2 = rx * rx + ry * ry + rz * rz;                                     // 5 FLOPs
    REALn rv = rx * vx + ry * vy + rz * vz;                                     // 5 FLOPs
    REALn v2 = vx * vx + vy * vy + vz * vz;                                     // 5 FLOPs
    REALn rj = rx * jx + ry * jy + rz * jz;                                     // 5 FLOPs
    REALn ra = rx * ax + ry * ay + rz * az;                                     // 5 FLOPs
    REALn va = vx * ax + vy * ay + vz * az;                                     // 5 FLOPs
    INTn mask = (r2 > 0);

    REALn inv_r2;
    REALn m_r3 = smoothed_m_r3_inv_r2(jm, r2, e2, mask, &inv_r2);               // 5 FLOPs

    REALn alpha = rv * inv_r2;                                                  // 1 FLOPs
    REALn alpha2 = alpha * alpha;                                               // 1 FLOPs
    REALn beta = 3 * ((v2 + ra) * inv_r2 + alpha2);                             // 4 FLOPs
    REALn gamma = (3 * va + rj) * inv_r2 + alpha * (beta - 4 * alpha2);         // 7 FLOPs

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
