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
// flop count: 59 + ???
{
	real_t r0x = irx - jrx;
	real_t r0y = iry - jry;
	real_t r0z = irz - jrz;
	real_t e2 = ie2 + je2;
	real_t v0x = ivx - jvx;
	real_t v0y = ivy - jvy;
	real_t v0z = ivz - jvz;
	real_t m = im + jm;

	real_t inv_m = 1 / m;
	real_t imu = im * inv_m;
	real_t jmu = jm * inv_m;

	real_t rcmx = imu * irx + jmu * jrx;
	real_t rcmy = imu * iry + jmu * jry;
	real_t rcmz = imu * irz + jmu * jrz;
	real_t vcmx = imu * ivx + jmu * jvx;
	real_t vcmy = imu * ivy + jmu * jvy;
	real_t vcmz = imu * ivz + jmu * jvz;

	rcmx += vcmx * dt;
	rcmy += vcmy * dt;
	rcmz += vcmz * dt;

	real_t r1x, r1y, r1z;
	real_t v1x, v1y, v1z;
	universal_kepler_solver(
		dt, m, e2,
		r0x, r0y, r0z,
		v0x, v0y, v0z,
		&r1x, &r1y, &r1z,
		&v1x, &v1y, &v1z);	// flop count: ???

	*ir1x = rcmx + jmu * r1x;
	*ir1y = rcmy + jmu * r1y;
	*ir1z = rcmz + jmu * r1z;
	*iv1x = vcmx + jmu * v1x;
	*iv1y = vcmy + jmu * v1y;
	*iv1z = vcmz + jmu * v1z;

	*jr1x = rcmx - imu * r1x;
	*jr1y = rcmy - imu * r1y;
	*jr1z = rcmz - imu * r1z;
	*jv1x = vcmx - imu * v1x;
	*jv1y = vcmy - imu * v1y;
	*jv1z = vcmz - imu * v1z;
}


#endif	// __KEPLER_SOLVER_KERNEL_COMMON_H__
