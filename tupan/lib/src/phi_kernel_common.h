#ifndef __PHI_KERNEL_COMMON_H__
#define __PHI_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

static inline void phi_kernel_core(
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
    REALn rx = irx - jrx;                                                       // 1 FLOPs
    REALn ry = iry - jry;                                                       // 1 FLOPs
    REALn rz = irz - jrz;                                                       // 1 FLOPs
    REALn e2 = ie2 + je2;                                                       // 1 FLOPs
    REALn r2 = rx * rx + ry * ry + rz * rz;                                     // 5 FLOPs
    INTn mask = (r2 > 0);

    REALn m_r1 = smoothed_m_r1(jm, r2, e2, mask);                               // 4 FLOPs

    *iphi -= m_r1;                                                              // 1 FLOPs
}
// Total flop count: 14

#endif  // __PHI_KERNEL_COMMON_H__
