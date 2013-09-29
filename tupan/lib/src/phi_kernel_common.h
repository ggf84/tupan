#ifndef __PHI_KERNEL_COMMON_H__
#define __PHI_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

inline void phi_kernel_core(
    const REALn im,
    const REALn irx,
    const REALn iry,
    const REALn irz,
    const REALn ie2,
    const REALn jm,
    const REALn jrx,
    const REALn jry,
    const REALn jrz,
    const REALn je2,
    REALn *iphi)
{
    REALn rx, ry, rz, e2;
    rx = irx - jrx;                                                             // 1 FLOPs
    ry = iry - jry;                                                             // 1 FLOPs
    rz = irz - jrz;                                                             // 1 FLOPs
    e2 = ie2 + je2;                                                             // 1 FLOPs
    REALn r2 = rx * rx + ry * ry + rz * rz;                                     // 5 FLOPs

    REALn inv_r1;
    smoothed_inv_r1(r2, e2, &inv_r1);                                           // 3 FLOPs

    *iphi -= jm * inv_r1;                                                       // 2 FLOPs
}
// Total flop count: 14

#endif  // __PHI_KERNEL_COMMON_H__
