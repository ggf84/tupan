#ifndef __SAKURA_KERNEL_COMMON_H__
#define __SAKURA_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"
#include "universal_kepler_solver.h"


static inline void twobody_solver(
    const REAL dt,
    const REAL m,
    const REAL e2,
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
    universal_kepler_solver(dt, m, e2, r0x, r0y, r0z, v0x, v0y, v0z,
                            &(*r1x), &(*r1y), &(*r1z),
                            &(*v1x), &(*v1y), &(*v1z));
    return;
}


static inline void evolve_twobody(
    const REAL dt,
    const INT flag,
    const REAL m,
    const REAL e2,
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

    if (flag == -1) {
        rx -= vx * dt;                                                              // 2 FLOPs
        ry -= vy * dt;                                                              // 2 FLOPs
        rz -= vz * dt;                                                              // 2 FLOPs
        twobody_solver(dt, m, e2, rx, ry, rz, vx, vy, vz,
                       &rx, &ry, &rz, &vx, &vy, &vz);                               // ? FLOPS
    }
    if (flag == 1) {
        twobody_solver(dt, m, e2, rx, ry, rz, vx, vy, vz,
                       &rx, &ry, &rz, &vx, &vy, &vz);                               // ? FLOPS
        rx -= vx * dt;                                                              // 2 FLOPs
        ry -= vy * dt;                                                              // 2 FLOPs
        rz -= vz * dt;                                                              // 2 FLOPs
    }
    if (flag == -2) {
        rx -= vx * dt / 2;                                                          // 2 FLOPs
        ry -= vy * dt / 2;                                                          // 2 FLOPs
        rz -= vz * dt / 2;                                                          // 2 FLOPs
        twobody_solver(dt, m, e2, rx, ry, rz, vx, vy, vz,
                       &rx, &ry, &rz, &vx, &vy, &vz);                               // ? FLOPS
        rx -= vx * dt / 2;                                                          // 2 FLOPs
        ry -= vy * dt / 2;                                                          // 2 FLOPs
        rz -= vz * dt / 2;                                                          // 2 FLOPs
    }
    if (flag == 2) {
        twobody_solver(dt/2, m, e2, rx, ry, rz, vx, vy, vz,
                       &rx, &ry, &rz, &vx, &vy, &vz);                               // ? FLOPS
        rx -= vx * dt;                                                              // 2 FLOPs
        ry -= vy * dt;                                                              // 2 FLOPs
        rz -= vz * dt;                                                              // 2 FLOPs
        twobody_solver(dt/2, m, e2, rx, ry, rz, vx, vy, vz,
                       &rx, &ry, &rz, &vx, &vy, &vz);                               // ? FLOPS
    }

    *r1x = rx;
    *r1y = ry;
    *r1z = rz;
    *v1x = vx;
    *v1y = vy;
    *v1z = vz;
}


static inline void sakura_kernel_core(
    const REAL dt,
    const INT flag,
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
    REAL r0x = irx - jrx;                                                       // 1 FLOPs
    REAL r0y = iry - jry;                                                       // 1 FLOPs
    REAL r0z = irz - jrz;                                                       // 1 FLOPs
    REAL e2 = ie2 + je2;                                                        // 1 FLOPs
    REAL v0x = ivx - jvx;                                                       // 1 FLOPs
    REAL v0y = ivy - jvy;                                                       // 1 FLOPs
    REAL v0z = ivz - jvz;                                                       // 1 FLOPs
    REAL m = im + jm;                                                           // 1 FLOPs

    REAL r1x, r1y, r1z;
    REAL v1x, v1y, v1z;
    evolve_twobody(dt, flag, m, e2, r0x, r0y, r0z, v0x, v0y, v0z,
                   &r1x, &r1y, &r1z, &v1x, &v1y, &v1z);                         // ? FLOPs

    REAL jmu = jm / m;                                                          // 1 FLOPs

    *idrx += jmu * (r1x - r0x);                                                 // 3 FLOPs
    *idry += jmu * (r1y - r0y);                                                 // 3 FLOPs
    *idrz += jmu * (r1z - r0z);                                                 // 3 FLOPs
    *idvx += jmu * (v1x - v0x);                                                 // 3 FLOPs
    *idvy += jmu * (v1y - v0y);                                                 // 3 FLOPs
    *idvz += jmu * (v1z - v0z);                                                 // 3 FLOPs
}
// Total flop count: 36 + ?

#endif  // __SAKURA_KERNEL_COMMON_H__
