#ifndef SAKURA_KERNEL_COMMON_H
#define SAKURA_KERNEL_COMMON_H

#include "common.h"
#include "smoothing.h"
#include "universal_kepler_solver.h"


static inline REAL
get_phi(const REAL m,
        const REAL x,
        const REAL y,
        const REAL z,
        const REAL eps2)
{
    REAL r2 = x * x + y * y + z * z;                                 // 5 FLOPs
    REAL inv_r = smoothed_inv_r1(r2, eps2);                          // 3 FLOPs
    return m * inv_r;                                                // 1 FLOPs
}


static inline REAL3
get_acc(const REAL m,
        const REAL x,
        const REAL y,
        const REAL z,
        const REAL eps2)
{
    REAL r2 = x * x + y * y + z * z;                                 // 5 FLOPs
    REAL inv_r3 = smoothed_inv_r3(r2, eps2);                         // 4 FLOPs
    REAL m_r3 = m * inv_r3;                                          // 1 FLOPs
    REAL3 a;
    a.x = -m_r3 * x;                                                 // 1 FLOPs
    a.y = -m_r3 * y;                                                 // 1 FLOPs
    a.z = -m_r3 * z;                                                 // 1 FLOPs
    return a;
}


static inline void
leapfrog(const REAL dt,
         const REAL4 r0,
         const REAL4 v0,
         REAL4 *r1,
         REAL4 *v1)
{
    REAL4 r = r0;
    REAL4 v = v0;
    REAL dt_2 = dt / 2;

    r.x += v.x * dt_2;
    r.y += v.y * dt_2;
    r.z += v.z * dt_2;

    REAL3 a = get_acc(r.w, r.x, r.y, r.z, v.w);

    v.x += a.x * dt;
    v.y += a.y * dt;
    v.z += a.z * dt;

    r.x += v.x * dt_2;
    r.y += v.y * dt_2;
    r.z += v.z * dt_2;

    *r1 = r;
    *v1 = v;
}


static inline void
twobody_solver(const REAL dt,
               const REAL4 r0,
               const REAL4 v0,
               REAL4 *r1,
               REAL4 *v1)
{
//    leapfrog(dt, r0, v0, r1, v1);
    universal_kepler_solver(dt, r0, v0, r1, v1);
}


inline REAL8
sakura_kernel_core(REAL8 idrdv,
                   const REAL4 irm, const REAL4 ive,
                   const REAL4 jrm, const REAL4 jve,
                   const REAL dt)
{
    REAL4 r0;
    r0.x = irm.x - jrm.x;                                            // 1 FLOPs
    r0.y = irm.y - jrm.y;                                            // 1 FLOPs
    r0.z = irm.z - jrm.z;                                            // 1 FLOPs
    r0.w = irm.w + jrm.w;                                            // 1 FLOPs
    REAL4 v0;
    v0.x = ive.x - jve.x;                                            // 1 FLOPs
    v0.y = ive.y - jve.y;                                            // 1 FLOPs
    v0.z = ive.z - jve.z;                                            // 1 FLOPs
    v0.w = ive.w + jve.w;                                            // 1 FLOPs

    REAL dt_2 = dt / 2;                                              // 1 FLOPs

    REAL4 rr0 = r0;
    rr0.x -= v0.x * dt_2;                                            // 2 FLOPs
    rr0.y -= v0.y * dt_2;                                            // 2 FLOPs
    rr0.z -= v0.z * dt_2;                                            // 2 FLOPs
    REAL4 vv0 = v0;

    REAL4 rr1, vv1;
    twobody_solver(dt, rr0, vv0, &rr1, &vv1);                        // ? FLOPS

    REAL4 r1 = rr1;
    r1.x -= vv1.x * dt_2;                                            // 2 FLOPs
    r1.y -= vv1.y * dt_2;                                            // 2 FLOPs
    r1.z -= vv1.z * dt_2;                                            // 2 FLOPs
    REAL4 v1 = vv1;

    REAL muj = jrm.w / r0.w;                                         // 1 FLOPs

    idrdv.s0 += muj * (r1.x - r0.x);                                 // 3 FLOPs
    idrdv.s1 += muj * (r1.y - r0.y);                                 // 3 FLOPs
    idrdv.s2 += muj * (r1.z - r0.z);                                 // 3 FLOPs
    idrdv.s3  = 0;
    idrdv.s4 += muj * (v1.x - v0.x);                                 // 3 FLOPs
    idrdv.s5 += muj * (v1.y - v0.y);                                 // 3 FLOPs
    idrdv.s6 += muj * (v1.z - v0.z);                                 // 3 FLOPs
    idrdv.s7  = 0;

    return idrdv;
}
// Total flop count: 36 + ?

#endif  // SAKURA_KERNEL_COMMON_H
