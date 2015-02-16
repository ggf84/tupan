#ifndef __TSTEP_KERNEL_COMMON_H__
#define __TSTEP_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


static inline void
tstep_kernel_core(
    real_tn const eta,
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
    real_tn *iw2_a,
    real_tn *iw2_b)
{
    real_tn rx = irx - jrx;                                                     // 1 FLOPs
    real_tn ry = iry - jry;                                                     // 1 FLOPs
    real_tn rz = irz - jrz;                                                     // 1 FLOPs
    real_tn e2 = ie2 + je2;                                                     // 1 FLOPs
    real_tn vx = ivx - jvx;                                                     // 1 FLOPs
    real_tn vy = ivy - jvy;                                                     // 1 FLOPs
    real_tn vz = ivz - jvz;                                                     // 1 FLOPs
    real_tn m = im + jm;                                                        // 1 FLOPs
    real_tn r2 = rx * rx + ry * ry + rz * rz;                                   // 5 FLOPs
    real_tn rv = rx * vx + ry * vy + rz * vz;                                   // 5 FLOPs
    real_tn v2 = vx * vx + vy * vy + vz * vz;                                   // 5 FLOPs
    int_tn mask = (r2 > 0);

    real_tn inv_r2;
    real_tn m_r1 = smoothed_m_r1_inv_r2(m, r2, e2, mask, &inv_r2);              // 4 FLOPs

    real_tn a = (real_tn)(2);
    real_tn b = (1 + a / 2) * inv_r2;                                           // 3 FLOPs

    real_tn w2 = (v2 + a * m_r1) * inv_r2;                                      // 3 FLOPs
    real_tn gamma = (w2 + b * m_r1) * inv_r2;                                   // 3 FLOPs
    gamma *= (eta / sqrt(w2));                                                  // 3 FLOPs
    w2 -= gamma * rv;                                                           // 2 FLOPs

    w2 = select((real_tn)(0), w2, mask);

    *iw2_a += w2;                                                               // 1 FLOPs
    *iw2_b = fmax(w2, *iw2_b);
}
// Total flop count: 42


#endif  // __TSTEP_KERNEL_COMMON_H__
