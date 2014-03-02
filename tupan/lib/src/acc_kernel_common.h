#ifndef __ACC_KERNEL_COMMON_H__
#define __ACC_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

static inline void acc_kernel_core(
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
    REALn rx = irx - jrx;                                                       // 1 FLOPs
    REALn ry = iry - jry;                                                       // 1 FLOPs
    REALn rz = irz - jrz;                                                       // 1 FLOPs
    REALn e2 = ie2 + je2;                                                       // 1 FLOPs
    REALn r2 = rx * rx + ry * ry + rz * rz;                                     // 5 FLOPs
    INTn mask = (r2 > 0);

    REALn m_r3 = smoothed_m_r3(jm, r2, e2, mask);                               // 5 FLOPs

    *iax -= m_r3 * rx;                                                          // 2 FLOPs
    *iay -= m_r3 * ry;                                                          // 2 FLOPs
    *iaz -= m_r3 * rz;                                                          // 2 FLOPs
}
// Total flop count: 20

#endif  // __ACC_KERNEL_COMMON_H__
