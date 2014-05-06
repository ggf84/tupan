#ifndef __TSTEP_KERNEL_COMMON_H__
#define __TSTEP_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

static inline void tstep_kernel_core(
    const REALn eta,
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
    REALn *iw2_a,
    REALn *iw2_b)
{
    REALn rx = irx - jrx;                                                       // 1 FLOPs
    REALn ry = iry - jry;                                                       // 1 FLOPs
    REALn rz = irz - jrz;                                                       // 1 FLOPs
    REALn e2 = ie2 + je2;                                                       // 1 FLOPs
    REALn vx = ivx - jvx;                                                       // 1 FLOPs
    REALn vy = ivy - jvy;                                                       // 1 FLOPs
    REALn vz = ivz - jvz;                                                       // 1 FLOPs
    REALn m = im + jm;                                                          // 1 FLOPs
    REALn r2 = rx * rx + ry * ry + rz * rz;                                     // 5 FLOPs
    REALn rv = rx * vx + ry * vy + rz * vz;                                     // 5 FLOPs
    REALn v2 = vx * vx + vy * vy + vz * vz;                                     // 5 FLOPs
    INTn mask = (r2 > 0);

    REALn inv_r2;
    REALn m_r1 = smoothed_m_r1_inv_r2(m, r2, e2, mask, &inv_r2);                // 4 FLOPs

    REALn a = (REALn)(2);
    REALn b = (1 + a / 2) * inv_r2;                                             // 3 FLOPs

    REALn w2 = (v2 + a * m_r1) * inv_r2;                                        // 3 FLOPs
    REALn gamma = (w2 + b * m_r1) * inv_r2;                                     // 3 FLOPs
    gamma *= (eta / sqrt(w2));                                                  // 3 FLOPs
    w2 -= gamma * rv;                                                           // 2 FLOPs

    w2 = select((REALn)(0), w2, mask);

    *iw2_a += w2;                                                               // 1 FLOPs
    *iw2_b = fmax(w2, *iw2_b);
}
// Total flop count: 42

#endif  // __TSTEP_KERNEL_COMMON_H__
