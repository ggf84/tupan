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

    REALn inv_r3;
    smoothed_inv_r3(r2, e2, mask, &inv_r3);                                     // 4 FLOPs

    inv_r3 *= jm;                                                               // 1 FLOPs

    *iax -= inv_r3 * rx;                                                        // 2 FLOPs
    *iay -= inv_r3 * ry;                                                        // 2 FLOPs
    *iaz -= inv_r3 * rz;                                                        // 2 FLOPs
}
// Total flop count: 20





#define IMPLEMENT_ACC_KERNEL_CORE(vw)                                                   \
                                                                                        \
    static inline void acc_kernel_core_##vw(                                            \
        const REAL##vw im,                                                              \
        const REAL##vw irx,                                                             \
        const REAL##vw iry,                                                             \
        const REAL##vw irz,                                                             \
        const REAL##vw ie2,                                                             \
        const REAL##vw jm,                                                              \
        const REAL##vw jrx,                                                             \
        const REAL##vw jry,                                                             \
        const REAL##vw jrz,                                                             \
        const REAL##vw je2,                                                             \
        REAL##vw *iax,                                                                  \
        REAL##vw *iay,                                                                  \
        REAL##vw *iaz)                                                                  \
    {                                                                                   \
        REAL##vw rx = irx - jrx;                                                        \
        REAL##vw ry = iry - jry;                                                        \
        REAL##vw rz = irz - jrz;                                                        \
        REAL##vw e2 = ie2 + je2;                                                        \
                                                                                        \
        REAL##vw r2 = rx * rx + ry * ry + rz * rz;                                      \
        INT##vw mask = (r2 > 0);                                                        \
                                                                                        \
        REAL##vw inv_r2 = 1 / (r2 + e2);                                                \
        inv_r2 = select((REAL##vw)(0), inv_r2, mask);                                   \
        REAL##vw inv_r3 = jm * inv_r2;                                                  \
        inv_r3 *= sqrt(inv_r2);                                                         \
                                                                                        \
        *iax -= inv_r3 * rx;                                                            \
        *iay -= inv_r3 * ry;                                                            \
        *iaz -= inv_r3 * rz;                                                            \
    }



IMPLEMENT_ACC_KERNEL_CORE(1)
IMPLEMENT_ACC_KERNEL_CORE(2)
IMPLEMENT_ACC_KERNEL_CORE(4)
IMPLEMENT_ACC_KERNEL_CORE(8)
IMPLEMENT_ACC_KERNEL_CORE(16)


#define call(func, vw) concat(func, concat(_, vw))


#endif  // __ACC_KERNEL_COMMON_H__
