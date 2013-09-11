#ifndef __PHI_KERNEL_COMMON_H__
#define __PHI_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

inline void phi_kernel_core(
    const REAL im,
    const REAL irx,
    const REAL iry,
    const REAL irz,
    const REAL ie2,
    const REAL jm,
    const REAL jrx,
    const REAL jry,
    const REAL jrz,
    const REAL je2,
    REAL *iphi)
{
    REAL rx, ry, rz;
    rx = irx - jrx;                                                             // 1 FLOPs
    ry = iry - jry;                                                             // 1 FLOPs
    rz = irz - jrz;                                                             // 1 FLOPs
    REAL r2 = rx * rx + ry * ry + rz * rz;                                      // 5 FLOPs

    REAL e2 = ie2 + je2;                                                        // 1 FLOPs

    REAL inv_r1;
    smoothed_inv_r1(r2, e2, &inv_r1);                                           // 3 FLOPs

    *iphi -= jm * inv_r1;                                                       // 2 FLOPs
}
// Total flop count: 14

#endif  // __PHI_KERNEL_COMMON_H__
