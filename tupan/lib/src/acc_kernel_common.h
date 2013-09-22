#ifndef __ACC_KERNEL_COMMON_H__
#define __ACC_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

inline void acc_kernel_core(
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
    REALn *iax,
    REALn *iay,
    REALn *iaz)
{
    REALn rx, ry, rz;
    rx = irx - jrx;                                                             // 1 FLOPs
    ry = iry - jry;                                                             // 1 FLOPs
    rz = irz - jrz;                                                             // 1 FLOPs
    REALn r2 = rx * rx + ry * ry + rz * rz;                                     // 5 FLOPs

    REALn e2 = ie2 + je2;                                                       // 1 FLOPs

    REALn inv_r3;
    smoothed_inv_r3(r2, e2, &inv_r3);                                           // 4 FLOPs

    inv_r3 *= jm;                                                               // 1 FLOPs

    *iax -= inv_r3 * rx;                                                        // 2 FLOPs
    *iay -= inv_r3 * ry;                                                        // 2 FLOPs
    *iaz -= inv_r3 * rz;                                                        // 2 FLOPs
}
// Total flop count: 20

#endif  // __ACC_KERNEL_COMMON_H__
