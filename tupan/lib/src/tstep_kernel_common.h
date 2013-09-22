#ifndef __TSTEP_KERNEL_COMMON_H__
#define __TSTEP_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

inline void tstep_kernel_core(
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
    REALn rx, ry, rz;
    rx = irx - jrx;                                                             // 1 FLOPs
    ry = iry - jry;                                                             // 1 FLOPs
    rz = irz - jrz;                                                             // 1 FLOPs
    REALn vx, vy, vz;
    vx = ivx - jvx;                                                             // 1 FLOPs
    vy = ivy - jvy;                                                             // 1 FLOPs
    vz = ivz - jvz;                                                             // 1 FLOPs
    REALn r2 = rx * rx + ry * ry + rz * rz;                                     // 5 FLOPs
    REALn rv = rx * vx + ry * vy + rz * vz;                                     // 5 FLOPs
    REALn v2 = vx * vx + vy * vy + vz * vz;                                     // 5 FLOPs

    REALn m = im + jm;                                                          // 1 FLOPs
    REALn e2 = ie2 + je2;                                                       // 1 FLOPs

    REALn inv_r1, inv_r2;
    smoothed_inv_r1r2(r2, e2, &inv_r1, &inv_r2);                                // 3 FLOPs

    REALn a = inv_r2;
    REALn b = 2 * inv_r2;                                                       // 1 FLOPs
    REALn c = (a + b / 2);                                                      // 2 FLOPs

    REALn phi = m * inv_r1;                                                     // 1 FLOPs
    REALn w2 = a * v2 + b * phi;                                                // 3 FLOPs
    REALn gamma = w2 + c * phi;                                                 // 2 FLOPs
    REALn dln_w = gamma * rv * inv_r2;                                          // 2 FLOPs
    w2 -= (eta / sqrt(w2)) * dln_w;                                             // 4 FLOPs

    w2 = select((REALn)(0), w2, (INTn)(r2 > 0));

    *iw2_a += w2;                                                               // 1 FLOPs
    *iw2_b = max(w2, *iw2_b);
}
// Total flop count: 42

#endif  // __TSTEP_KERNEL_COMMON_H__
