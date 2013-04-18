#ifndef SAKURA_KERNEL_COMMON_H
#define SAKURA_KERNEL_COMMON_H

#include "common.h"
#include "smoothing.h"
#include "universal_kepler_solver.h"


static inline REAL
get_phi(
    const REAL m,
    const REAL x,
    const REAL y,
    const REAL z,
    const REAL eps2
    )
{
    REAL r2 = x * x + y * y + z * z;                                 // 5 FLOPs
    REAL inv_r = smoothed_inv_r1(r2, eps2);                          // 3 FLOPs
    return m * inv_r;                                                // 1 FLOPs
}


static inline REAL3
get_acc(
    const REAL m,
    const REAL x,
    const REAL y,
    const REAL z,
    const REAL eps2
    )
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
leapfrog(
    const REAL dt,
    const REAL4 r0,
    const REAL4 v0,
    REAL4 *r1,
    REAL4 *v1
    )
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
twobody_solver(
    const REAL dt,
    const REAL4 r0,
    const REAL4 v0,
    REAL4 *r1,
    REAL4 *v1
    )
{
//    leapfrog(dt, r0, v0, r1, v1);
    universal_kepler_solver(dt, r0, v0, r1, v1);
}


static inline void
evolve_twobody(
    const REAL dt,
    const REAL4 r0,
    const REAL4 v0,
    REAL4 *r1,
    REAL4 *v1
    )
{
    REAL4 r = r0;
    REAL4 v = v0;
    REAL dt_2 = dt / 2;                                              // 1 FLOPs

    r.x -= v.x * dt_2;                                               // 2 FLOPs
    r.y -= v.y * dt_2;                                               // 2 FLOPs
    r.z -= v.z * dt_2;                                               // 2 FLOPs

    twobody_solver(dt, r, v, &r, &v);                                // ? FLOPS

    r.x -= v.x * dt_2;                                               // 2 FLOPs
    r.y -= v.y * dt_2;                                               // 2 FLOPs
    r.z -= v.z * dt_2;                                               // 2 FLOPs

    *r1 = r;
    *v1 = v;
}


inline void
sakura_kernel_core(
    const REAL dt,
    const REAL4 irm, const REAL4 ive,
    const REAL4 jrm, const REAL4 jve,
    REAL3 *idr,
    REAL3 *idv
    )
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

    REAL4 r1, v1;
    evolve_twobody(dt, r0, v0, &r1, &v1);                            // ? FLOPs

    REAL muj = jrm.w / r0.w;                                         // 1 FLOPs

    idr->x += muj * (r1.x - r0.x);                                    // 3 FLOPs
    idr->y += muj * (r1.y - r0.y);                                    // 3 FLOPs
    idr->z += muj * (r1.z - r0.z);                                    // 3 FLOPs
    idv->x += muj * (v1.x - v0.x);                                    // 3 FLOPs
    idv->y += muj * (v1.y - v0.y);                                    // 3 FLOPs
    idv->z += muj * (v1.z - v0.z);                                    // 3 FLOPs
}
// Total flop count: 36 + ?

#endif  // SAKURA_KERNEL_COMMON_H
