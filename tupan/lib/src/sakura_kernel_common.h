#ifndef __SAKURA_KERNEL_COMMON_H__
#define __SAKURA_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"
#include "universal_kepler_solver.h"


static inline void
twobody_solver(
    REAL const dt,
    REAL const m,
    REAL const e2,
    REAL const r0x,
    REAL const r0y,
    REAL const r0z,
    REAL const v0x,
    REAL const v0y,
    REAL const v0z,
    REAL *r1x,
    REAL *r1y,
    REAL *r1z,
    REAL *v1x,
    REAL *v1y,
    REAL *v1z)
{
    universal_kepler_solver(dt, m, e2,
                            r0x, r0y, r0z,
                            v0x, v0y, v0z,
                            &(*r1x), &(*r1y), &(*r1z),
                            &(*v1x), &(*v1y), &(*v1z));
}


static inline void
evolve_twobody(
    REAL const dt,
    INT const flag,
    REAL const m,
    REAL const e2,
    REAL const r0x,
    REAL const r0y,
    REAL const r0z,
    REAL const v0x,
    REAL const v0y,
    REAL const v0z,
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
        rx -= vx * dt;                                                          // 2 FLOPs
        ry -= vy * dt;                                                          // 2 FLOPs
        rz -= vz * dt;                                                          // 2 FLOPs
        twobody_solver(dt, m, e2, rx, ry, rz, vx, vy, vz,
                       &rx, &ry, &rz, &vx, &vy, &vz);                           // ? FLOPS
    }
    if (flag == 1) {
        twobody_solver(dt, m, e2, rx, ry, rz, vx, vy, vz,
                       &rx, &ry, &rz, &vx, &vy, &vz);                           // ? FLOPS
        rx -= vx * dt;                                                          // 2 FLOPs
        ry -= vy * dt;                                                          // 2 FLOPs
        rz -= vz * dt;                                                          // 2 FLOPs
    }
    if (flag == -2) {
        rx -= vx * dt / 2;                                                      // 2 FLOPs
        ry -= vy * dt / 2;                                                      // 2 FLOPs
        rz -= vz * dt / 2;                                                      // 2 FLOPs
        twobody_solver(dt, m, e2, rx, ry, rz, vx, vy, vz,
                       &rx, &ry, &rz, &vx, &vy, &vz);                           // ? FLOPS
        rx -= vx * dt / 2;                                                      // 2 FLOPs
        ry -= vy * dt / 2;                                                      // 2 FLOPs
        rz -= vz * dt / 2;                                                      // 2 FLOPs
    }
    if (flag == 2) {
        twobody_solver(dt/2, m, e2, rx, ry, rz, vx, vy, vz,
                       &rx, &ry, &rz, &vx, &vy, &vz);                           // ? FLOPS
        rx -= vx * dt;                                                          // 2 FLOPs
        ry -= vy * dt;                                                          // 2 FLOPs
        rz -= vz * dt;                                                          // 2 FLOPs
        twobody_solver(dt/2, m, e2, rx, ry, rz, vx, vy, vz,
                       &rx, &ry, &rz, &vx, &vy, &vz);                           // ? FLOPS
    }

    *r1x = rx;
    *r1y = ry;
    *r1z = rz;
    *v1x = vx;
    *v1y = vy;
    *v1z = vz;
}


static inline void
sakura_kernel_core(
    REAL const dt,
    INT const flag,
    REAL const im,
    REAL const irx,
    REAL const iry,
    REAL const irz,
    REAL const ie2,
    REAL const ivx,
    REAL const ivy,
    REAL const ivz,
    REAL const jm,
    REAL const jrx,
    REAL const jry,
    REAL const jrz,
    REAL const je2,
    REAL const jvx,
    REAL const jvy,
    REAL const jvz,
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
