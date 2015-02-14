#ifndef __PNACC_KERNEL_COMMON_H__
#define __PNACC_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"
#include "pn_terms.h"


static inline void
pnacc_kernel_core(
    REALn const im,
    REALn const irx,
    REALn const iry,
    REALn const irz,
    REALn const ie2,
    REALn const ivx,
    REALn const ivy,
    REALn const ivz,
    REALn const jm,
    REALn const jrx,
    REALn const jry,
    REALn const jrz,
    REALn const je2,
    REALn const jvx,
    REALn const jvy,
    REALn const jvz,
    CLIGHT const clight,
    REALn *ipnax,
    REALn *ipnay,
    REALn *ipnaz)
{
    REALn rx = irx - jrx;                                                       // 1 FLOPs
    REALn ry = iry - jry;                                                       // 1 FLOPs
    REALn rz = irz - jrz;                                                       // 1 FLOPs
    REALn e2 = ie2 + je2;                                                       // 1 FLOPs
    REALn vx = ivx - jvx;                                                       // 1 FLOPs
    REALn vy = ivy - jvy;                                                       // 1 FLOPs
    REALn vz = ivz - jvz;                                                       // 1 FLOPs
//    REALn m = im + jm;                                                          // 1 FLOPs
    REALn r2 = rx * rx + ry * ry + rz * rz;                                     // 5 FLOPs
    REALn v2 = vx * vx + vy * vy + vz * vz;                                     // 5 FLOPs
    INTn mask = (r2 > 0);

    REALn inv_r2;
    REALn inv_r1 = smoothed_inv_r1r2(r2, e2, mask, &inv_r2);                    // 3 FLOPs

    REALn nx = rx * inv_r1;                                                     // 1 FLOPs
    REALn ny = ry * inv_r1;                                                     // 1 FLOPs
    REALn nz = rz * inv_r1;                                                     // 1 FLOPs

//    REALn r_sch = 2 * m * clight.inv2;                                          // 2 FLOPs
//    REALn gamma2_a = r_sch * inv_r1;                                            // 1 FLOPs
//    REALn gamma2_b = v2 * clight.inv2;                                          // 1 FLOPs
//    REALn gamma2 = gamma2_a + gamma2_b;                                         // 1 FLOPs

    PN pn = PN_Init(0, 0);
    /* PN acceleration will only be calculated
     * if the condition below is fulfilled:
     * since gamma ~ v/c > 0.1% = 1e-3 therefore
     * gamma2 > 1e-6 should be our condition.
     */
//    INTn mask = (gamma2 > (REALn)(1.0e-6));
//    if (any(mask)) {
        p2p_pnterms(im, jm,
                    nx, ny, nz, vx, vy, vz, v2,
                    ivx, ivy, ivz, jvx, jvy, jvz,
                    inv_r1, inv_r2, clight, &pn);                               // ??? FLOPs
//        pn.a = select((REALn)(0), pn.a, mask);
//        pn.b = select((REALn)(0), pn.b, mask);
//    }
    *ipnax += pn.a * nx + pn.b * vx;                                            // 4 FLOPs
    *ipnay += pn.a * ny + pn.b * vy;                                            // 4 FLOPs
    *ipnaz += pn.a * nz + pn.b * vz;                                            // 4 FLOPs
}
// Total flop count: 40+???


#endif  // __PNACC_KERNEL_COMMON_H__
