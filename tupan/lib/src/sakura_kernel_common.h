#ifndef __SAKURA_KERNEL_COMMON_H__
#define __SAKURA_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"
#include "universal_kepler_solver.h"

inline void get_phi(
    const REAL m,
    const REAL e2,
    const REAL rx,
    const REAL ry,
    const REAL rz,
    REAL *phi)
{
    REAL r2 = rx * rx + ry * ry + rz * rz;                                      // 5 FLOPs
    REAL inv_r1;
    smoothed_inv_r1(r2, e2, &inv_r1);                                           // 3 FLOPs
    *phi = m * inv_r1;                                                          // 1 FLOPs
}


inline void get_acc(
    const REAL m,
    const REAL e2,
    const REAL rx,
    const REAL ry,
    const REAL rz,
    REAL *ax,
    REAL *ay,
    REAL *az)
{
    REAL r2 = rx * rx + ry * ry + rz * rz;                                      // 5 FLOPs
    REAL inv_r3;
    smoothed_inv_r3(r2, e2, &inv_r3);                                           // 4 FLOPs
    REAL m_r3 = m * inv_r3;                                                     // 1 FLOPs
    *ax = -m_r3 * rx;                                                           // 1 FLOPs
    *ay = -m_r3 * ry;                                                           // 1 FLOPs
    *az = -m_r3 * rz;                                                           // 1 FLOPs
}


inline void leapfrog(
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
    REAL dt_2 = dt / 2;
    REAL rx = r0x;
    REAL ry = r0y;
    REAL rz = r0z;
    REAL vx = v0x;
    REAL vy = v0y;
    REAL vz = v0z;

    rx += vx * dt_2;
    ry += vy * dt_2;
    rz += vz * dt_2;

    REAL ax, ay, az;
    get_acc(m, e2, rx, ry, rz, &ax, &ay, &az);

    vx += ax * dt;
    vy += ay * dt;
    vz += az * dt;

    rx += vx * dt_2;
    ry += vy * dt_2;
    rz += vz * dt_2;

    *r1x = rx;
    *r1y = ry;
    *r1z = rz;
    *v1x = vx;
    *v1y = vy;
    *v1z = vz;
}


inline void twobody_solver(
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
    REAL r2 = r0x * r0x + r0y * r0y + r0z * r0z;
    REAL v2 = v0x * v0x + v0y * v0y + v0z * v0z;
    REAL R = 64 * (m / v2);
    if (r2 > R*R) {
    leapfrog(dt, m, e2, r0x, r0y, r0z, v0x, v0y, v0z,
             &(*r1x), &(*r1y), &(*r1z),
             &(*v1x), &(*v1y), &(*v1z));
    } else {
    universal_kepler_solver(dt, m, e2, r0x, r0y, r0z, v0x, v0y, v0z,
                            &(*r1x), &(*r1y), &(*r1z),
                            &(*v1x), &(*v1y), &(*v1z));
    }
}


inline void evolve_twobody(
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


inline void sakura_kernel_core(
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
    REAL r0x, r0y, r0z;
    r0x = irx - jrx;                                                            // 1 FLOPs
    r0y = iry - jry;                                                            // 1 FLOPs
    r0z = irz - jrz;                                                            // 1 FLOPs
    REAL v0x, v0y, v0z;
    v0x = ivx - jvx;                                                            // 1 FLOPs
    v0y = ivy - jvy;                                                            // 1 FLOPs
    v0z = ivz - jvz;                                                            // 1 FLOPs

    REAL m = im + jm;                                                           // 1 FLOPs
    REAL e2 = ie2 + je2;                                                        // 1 FLOPs

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
