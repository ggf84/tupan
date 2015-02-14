#ifndef __KEPLER_SOLVER_KERNEL_COMMON_H__
#define __KEPLER_SOLVER_KERNEL_COMMON_H__

#include "common.h"
#include "universal_kepler_solver.h"


static inline void
kepler_solver_kernel_core(
    REAL const dt,
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
    REAL *ir1x,
    REAL *ir1y,
    REAL *ir1z,
    REAL *iv1x,
    REAL *iv1y,
    REAL *iv1z,
    REAL *jr1x,
    REAL *jr1y,
    REAL *jr1z,
    REAL *jv1x,
    REAL *jv1y,
    REAL *jv1z)
{
    REAL r0x = irx - jrx;                                                       // 1 FLOPs
    REAL r0y = iry - jry;                                                       // 1 FLOPs
    REAL r0z = irz - jrz;                                                       // 1 FLOPs
    REAL e2 = ie2 + je2;                                                        // 1 FLOPs
    REAL v0x = ivx - jvx;                                                       // 1 FLOPs
    REAL v0y = ivy - jvy;                                                       // 1 FLOPs
    REAL v0z = ivz - jvz;                                                       // 1 FLOPs
    REAL m = im + jm;                                                           // 1 FLOPs

    REAL inv_m = 1 / m;                                                         // 1 FLOPs
    REAL imu = im * inv_m;                                                      // 1 FLOPs
    REAL jmu = jm * inv_m;                                                      // 1 FLOPs

    REAL rcmx = imu * irx + jmu * jrx;                                          // 3 FLOPs
    REAL rcmy = imu * iry + jmu * jry;                                          // 3 FLOPs
    REAL rcmz = imu * irz + jmu * jrz;                                          // 3 FLOPs
    REAL vcmx = imu * ivx + jmu * jvx;                                          // 3 FLOPs
    REAL vcmy = imu * ivy + jmu * jvy;                                          // 3 FLOPs
    REAL vcmz = imu * ivz + jmu * jvz;                                          // 3 FLOPs

    rcmx += vcmx * dt;                                                          // 2 FLOPs
    rcmy += vcmy * dt;                                                          // 2 FLOPs
    rcmz += vcmz * dt;                                                          // 2 FLOPs

    REAL r1x, r1y, r1z;
    REAL v1x, v1y, v1z;
    universal_kepler_solver(dt, m, e2,
                            r0x, r0y, r0z,
                            v0x, v0y, v0z,
                            &r1x, &r1y, &r1z,
                            &v1x, &v1y, &v1z);                                  // ? FLOPs

    *ir1x = rcmx + jmu * r1x;                                                   // 2 FLOPs
    *ir1y = rcmy + jmu * r1y;                                                   // 2 FLOPs
    *ir1z = rcmz + jmu * r1z;                                                   // 2 FLOPs
    *iv1x = vcmx + jmu * v1x;                                                   // 2 FLOPs
    *iv1y = vcmy + jmu * v1y;                                                   // 2 FLOPs
    *iv1z = vcmz + jmu * v1z;                                                   // 2 FLOPs

    *jr1x = rcmx - imu * r1x;                                                   // 2 FLOPs
    *jr1y = rcmy - imu * r1y;                                                   // 2 FLOPs
    *jr1z = rcmz - imu * r1z;                                                   // 2 FLOPs
    *jv1x = vcmx - imu * v1x;                                                   // 2 FLOPs
    *jv1y = vcmy - imu * v1y;                                                   // 2 FLOPs
    *jv1z = vcmz - imu * v1z;                                                   // 2 FLOPs
}
// Total flop count: 59 + ?


#endif  // __KEPLER_SOLVER_KERNEL_COMMON_H__
