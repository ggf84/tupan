#ifndef __PNACC_KERNEL_COMMON_H__
#define __PNACC_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"
#include "pn_terms.h"

inline void pnacc_kernel_core(
    const REAL im,
    const REAL irx,
    const REAL iry,
    const REAL irz,
    const REAL ie2,
    const REAL ivx,
    const REAL ivy,
    const REAL ivz,
    const REAL jm,
    const REAL jrx,
    const REAL jry,
    const REAL jrz,
    const REAL je2,
    const REAL jvx,
    const REAL jvy,
    const REAL jvz,
    const CLIGHT clight,
    REAL *ipnax,
    REAL *ipnay,
    REAL *ipnaz)
{
    REAL rx, ry, rz, e2;
    rx = irx - jrx;                                                             // 1 FLOPs
    ry = iry - jry;                                                             // 1 FLOPs
    rz = irz - jrz;                                                             // 1 FLOPs
    e2 = ie2 + je2;                                                             // 1 FLOPs
    REAL vx, vy, vz, m;
    vx = ivx - jvx;                                                             // 1 FLOPs
    vy = ivy - jvy;                                                             // 1 FLOPs
    vz = ivz - jvz;                                                             // 1 FLOPs
    m = im + jm;                                                                // 1 FLOPs
    REAL r2 = rx * rx + ry * ry + rz * rz;                                      // 5 FLOPs
    REAL v2 = vx * vx + vy * vy + vz * vz;                                      // 5 FLOPs

    REAL inv_r1, inv_r2, inv_r3;
    smoothed_inv_r1r2r3(r2, e2, &inv_r1, &inv_r2, &inv_r3);                     // 4 FLOPs

    REAL r_sch = 2 * m * clight.inv2;                                           // 2 FLOPs
    REAL gamma2_a = r_sch * inv_r1;                                             // 1 FLOPs
    REAL gamma2_b = v2 * clight.inv2;                                           // 1 FLOPs
    REAL gamma2 = gamma2_a + gamma2_b;                                          // 1 FLOPs

    PN pn = {0, 0};
    /* PN acceleration will only be calculated
     * if the condition below is fulfilled:
     * since gamma ~ v/c > 0.1% = 1e-3 therefore
     * gamma2 > 1e-6 should be our condition.
     */
    if (1e6*gamma2 > 1) {
        p2p_pnterms(im, jm,
                    rx, ry, rz, vx, vy, vz, v2,
                    ivx, ivy, ivz, jvx, jvy, jvz,
                    inv_r1, inv_r2, inv_r3,
                    clight, &pn);                                               // ??? FLOPs
    }
    *ipnax += pn.a * rx + pn.b * vx;                                            // 4 FLOPs
    *ipnay += pn.a * ry + pn.b * vy;                                            // 4 FLOPs
    *ipnaz += pn.a * rz + pn.b * vz;                                            // 4 FLOPs
}
// Total flop count: 40+???

#endif  // __PNACC_KERNEL_COMMON_H__
