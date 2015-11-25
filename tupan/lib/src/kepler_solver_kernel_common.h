#ifndef __KEPLER_SOLVER_KERNEL_COMMON_H__
#define __KEPLER_SOLVER_KERNEL_COMMON_H__

#include "common.h"
#include "universal_kepler_solver.h"


static inline void
kepler_solver_kernel_core(
	const real_t dt,
	const real_t im,
	const real_t irx,
	const real_t iry,
	const real_t irz,
	const real_t ie2,
	const real_t ivx,
	const real_t ivy,
	const real_t ivz,
	const real_t jm,
	const real_t jrx,
	const real_t jry,
	const real_t jrz,
	const real_t je2,
	const real_t jvx,
	const real_t jvy,
	const real_t jvz,
	real_t *ir1x,
	real_t *ir1y,
	real_t *ir1z,
	real_t *iv1x,
	real_t *iv1y,
	real_t *iv1z,
	real_t *jr1x,
	real_t *jr1y,
	real_t *jr1z,
	real_t *jv1x,
	real_t *jv1y,
	real_t *jv1z)
{
	real_t r0x = irx - jrx;														// 1 FLOPs
	real_t r0y = iry - jry;														// 1 FLOPs
	real_t r0z = irz - jrz;														// 1 FLOPs
	real_t e2 = ie2 + je2;														// 1 FLOPs
	real_t v0x = ivx - jvx;														// 1 FLOPs
	real_t v0y = ivy - jvy;														// 1 FLOPs
	real_t v0z = ivz - jvz;														// 1 FLOPs
	real_t m = im + jm;															// 1 FLOPs

	real_t inv_m = 1 / m;														// 1 FLOPs
	real_t imu = im * inv_m;													// 1 FLOPs
	real_t jmu = jm * inv_m;													// 1 FLOPs

	real_t rcmx = imu * irx + jmu * jrx;										// 3 FLOPs
	real_t rcmy = imu * iry + jmu * jry;										// 3 FLOPs
	real_t rcmz = imu * irz + jmu * jrz;										// 3 FLOPs
	real_t vcmx = imu * ivx + jmu * jvx;										// 3 FLOPs
	real_t vcmy = imu * ivy + jmu * jvy;										// 3 FLOPs
	real_t vcmz = imu * ivz + jmu * jvz;										// 3 FLOPs

	rcmx += vcmx * dt;															// 2 FLOPs
	rcmy += vcmy * dt;															// 2 FLOPs
	rcmz += vcmz * dt;															// 2 FLOPs

	real_t r1x, r1y, r1z;
	real_t v1x, v1y, v1z;
	universal_kepler_solver(
		dt, m, e2,
		r0x, r0y, r0z,
		v0x, v0y, v0z,
		&r1x, &r1y, &r1z,
		&v1x, &v1y, &v1z);														// ? FLOPs

	*ir1x = rcmx + jmu * r1x;													// 2 FLOPs
	*ir1y = rcmy + jmu * r1y;													// 2 FLOPs
	*ir1z = rcmz + jmu * r1z;													// 2 FLOPs
	*iv1x = vcmx + jmu * v1x;													// 2 FLOPs
	*iv1y = vcmy + jmu * v1y;													// 2 FLOPs
	*iv1z = vcmz + jmu * v1z;													// 2 FLOPs

	*jr1x = rcmx - imu * r1x;													// 2 FLOPs
	*jr1y = rcmy - imu * r1y;													// 2 FLOPs
	*jr1z = rcmz - imu * r1z;													// 2 FLOPs
	*jv1x = vcmx - imu * v1x;													// 2 FLOPs
	*jv1y = vcmy - imu * v1y;													// 2 FLOPs
	*jv1z = vcmz - imu * v1z;													// 2 FLOPs
}
// Total flop count: 59 + ?


#endif	// __KEPLER_SOLVER_KERNEL_COMMON_H__
