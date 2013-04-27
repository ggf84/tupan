#ifndef __SAKURA_KERNEL_COMMON_H__
#define __SAKURA_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"
#include "universal_kepler_solver.h"

inline REAL
get_phi(
    const REAL m,
    const REAL x,
    const REAL y,
    const REAL z,
    const REAL eps2)
{
    REAL r2 = x * x + y * y + z * z;                                 // 5 FLOPs
    REAL inv_r1;
    smoothed_inv_r1(r2, eps2, &inv_r1);                              // 3 FLOPs
    return m * inv_r1;                                               // 1 FLOPs
}


inline REAL3
get_acc(
    const REAL m,
    const REAL x,
    const REAL y,
    const REAL z,
    const REAL eps2)
{
    REAL r2 = x * x + y * y + z * z;                                 // 5 FLOPs
    REAL inv_r3;
    smoothed_inv_r3(r2, eps2, &inv_r3);                              // 4 FLOPs
    REAL m_r3 = m * inv_r3;                                          // 1 FLOPs
    REAL3 a;
    a.x = -m_r3 * x;                                                 // 1 FLOPs
    a.y = -m_r3 * y;                                                 // 1 FLOPs
    a.z = -m_r3 * z;                                                 // 1 FLOPs
    return a;
}


inline void
leapfrog(
    const REAL dt,
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


inline void
twobody_solver(
    const REAL dt,
    const REAL m,
    const REAL r0x,
    const REAL r0y,
    const REAL r0z,
    const REAL v0x,
    const REAL v0y,
    const REAL v0z,
    REAL *r1x,
    REAL *r1y,
    REAL *r1z,
    REAL *v1x,
    REAL *v1y,
    REAL *v1z)
{
    REAL4 r0 = (REAL4){r0x, r0y, r0z, m};
    REAL4 v0 = (REAL4){v0x, v0y, v0z, 0};

    REAL4 r1 = r0;
    REAL4 v1 = v0;

//    leapfrog(dt, r0, v0, &r1, &v1);
    universal_kepler_solver(dt, r0, v0, &r1, &v1);

    *r1x = r1.x;
    *r1y = r1.y;
    *r1z = r1.z;
    *v1x = v1.x;
    *v1y = v1.y;
    *v1z = v1.z;
}


inline void
evolve_twobody(
    const REAL dt,
    const REAL m,
    const REAL r0x,
    const REAL r0y,
    const REAL r0z,
    const REAL v0x,
    const REAL v0y,
    const REAL v0z,
    REAL *r1x,
    REAL *r1y,
    REAL *r1z,
    REAL *v1x,
    REAL *v1y,
    REAL *v1z)
{
    REAL rx = r0x;
    REAL ry = r0y;
    REAL rz = r0z;
    REAL vx = v0x;
    REAL vy = v0y;
    REAL vz = v0z;
    REAL dt_2 = dt / 2;                                              // 1 FLOPs

    rx -= vx * dt_2;                                                 // 2 FLOPs
    ry -= vy * dt_2;                                                 // 2 FLOPs
    rz -= vz * dt_2;                                                 // 2 FLOPs

    twobody_solver(dt, m, rx, ry, rz, vx, vy, vz,
                   &rx, &ry, &rz, &vx, &vy, &vz);                    // ? FLOPS

    rx -= vx * dt_2;                                                 // 2 FLOPs
    ry -= vy * dt_2;                                                 // 2 FLOPs
    rz -= vz * dt_2;                                                 // 2 FLOPs

    *r1x = rx;
    *r1y = ry;
    *r1z = rz;
    *v1x = vx;
    *v1y = vy;
    *v1z = vz;
}


inline void
sakura_kernel_core(
    const REAL dt,
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
    REAL *idrx,
    REAL *idry,
    REAL *idrz,
    REAL *idvx,
    REAL *idvy,
    REAL *idvz)
{
    REAL r0x, r0y, r0z;
    r0x = irx - jrx;                                                 // 1 FLOPs
    r0y = iry - jry;                                                 // 1 FLOPs
    r0z = irz - jrz;                                                 // 1 FLOPs
    REAL v0x, v0y, v0z;
    v0x = ivx - jvx;                                                 // 1 FLOPs
    v0y = ivy - jvy;                                                 // 1 FLOPs
    v0z = ivz - jvz;                                                 // 1 FLOPs

    REAL m = im + jm;

    REAL r1x, r1y, r1z;
    REAL v1x, v1y, v1z;
    evolve_twobody(dt, m, r0x, r0y, r0z, v0x, v0y, v0z,
                   &r1x, &r1y, &r1z, &v1x, &v1y, &v1z);              // ? FLOPs

    REAL jmu = jm / m;                                               // 1 FLOPs

    *idrx += jmu * (r1x - r0x);                                      // 3 FLOPs
    *idry += jmu * (r1y - r0y);                                      // 3 FLOPs
    *idrz += jmu * (r1z - r0z);                                      // 3 FLOPs
    *idvx += jmu * (v1x - v0x);                                      // 3 FLOPs
    *idvy += jmu * (v1y - v0y);                                      // 3 FLOPs
    *idvz += jmu * (v1z - v0z);                                      // 3 FLOPs
}
// Total flop count: 36 + ?

#endif  // __SAKURA_KERNEL_COMMON_H__
