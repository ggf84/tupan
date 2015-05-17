#ifndef __SAKURA_KERNEL_COMMON_H__
#define __SAKURA_KERNEL_COMMON_H__

#include "common.h"
#include "universal_kepler_solver.h"


static inline void
twobody_solver(
	real_t const dt,
	real_t const m,
	real_t const e2,
	real_t const r0x,
	real_t const r0y,
	real_t const r0z,
	real_t const v0x,
	real_t const v0y,
	real_t const v0z,
	real_t *r1x,
	real_t *r1y,
	real_t *r1z,
	real_t *v1x,
	real_t *v1y,
	real_t *v1z)
{
	universal_kepler_solver(
		dt, m, e2,
		r0x, r0y, r0z,
		v0x, v0y, v0z,
		&(*r1x), &(*r1y), &(*r1z),
		&(*v1x), &(*v1y), &(*v1z));
}


static inline void
evolve_twobody(
	real_t const dt,
	int_t const flag,
	real_t const m,
	real_t const e2,
	real_t const r0x,
	real_t const r0y,
	real_t const r0z,
	real_t const v0x,
	real_t const v0y,
	real_t const v0z,
	real_t *r1x,
	real_t *r1y,
	real_t *r1z,
	real_t *v1x,
	real_t *v1y,
	real_t *v1z)
{
	real_t rx = r0x;
	real_t ry = r0y;
	real_t rz = r0z;
	real_t vx = v0x;
	real_t vy = v0y;
	real_t vz = v0z;

	if (flag == -1) {
		rx -= vx * dt;															// 2 FLOPs
		ry -= vy * dt;															// 2 FLOPs
		rz -= vz * dt;															// 2 FLOPs
		twobody_solver(
			dt, m, e2, rx, ry, rz, vx, vy, vz,
			&rx, &ry, &rz, &vx, &vy, &vz);										// ? FLOPS
	}
	if (flag == 1) {
		twobody_solver(
			dt, m, e2, rx, ry, rz, vx, vy, vz,
			&rx, &ry, &rz, &vx, &vy, &vz);										// ? FLOPS
		rx -= vx * dt;															// 2 FLOPs
		ry -= vy * dt;															// 2 FLOPs
		rz -= vz * dt;															// 2 FLOPs
	}
	if (flag == -2) {
		rx -= vx * dt / 2;														// 2 FLOPs
		ry -= vy * dt / 2;														// 2 FLOPs
		rz -= vz * dt / 2;														// 2 FLOPs
		twobody_solver(
			dt, m, e2, rx, ry, rz, vx, vy, vz,
			&rx, &ry, &rz, &vx, &vy, &vz);										// ? FLOPS
		rx -= vx * dt / 2;														// 2 FLOPs
		ry -= vy * dt / 2;														// 2 FLOPs
		rz -= vz * dt / 2;														// 2 FLOPs
	}
	if (flag == 2) {
		twobody_solver(
			dt/2, m, e2, rx, ry, rz, vx, vy, vz,
			&rx, &ry, &rz, &vx, &vy, &vz);										// ? FLOPS
		rx -= vx * dt;															// 2 FLOPs
		ry -= vy * dt;															// 2 FLOPs
		rz -= vz * dt;															// 2 FLOPs
		twobody_solver(
			dt/2, m, e2, rx, ry, rz, vx, vy, vz,
			&rx, &ry, &rz, &vx, &vy, &vz);										// ? FLOPS
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
	real_t const dt,
	int_t const flag,
	real_t const im,
	real_t const irx,
	real_t const iry,
	real_t const irz,
	real_t const ie2,
	real_t const ivx,
	real_t const ivy,
	real_t const ivz,
	real_t const jm,
	real_t const jrx,
	real_t const jry,
	real_t const jrz,
	real_t const je2,
	real_t const jvx,
	real_t const jvy,
	real_t const jvz,
	real_t *idrx,
	real_t *idry,
	real_t *idrz,
	real_t *idvx,
	real_t *idvy,
	real_t *idvz)
{
	real_t r0x = irx - jrx;														// 1 FLOPs
	real_t r0y = iry - jry;														// 1 FLOPs
	real_t r0z = irz - jrz;														// 1 FLOPs
	real_t e2 = ie2 + je2;														// 1 FLOPs
	real_t v0x = ivx - jvx;														// 1 FLOPs
	real_t v0y = ivy - jvy;														// 1 FLOPs
	real_t v0z = ivz - jvz;														// 1 FLOPs
	real_t m = im + jm;															// 1 FLOPs

	real_t r1x, r1y, r1z;
	real_t v1x, v1y, v1z;
	evolve_twobody(
		dt, flag, m, e2, r0x, r0y, r0z, v0x, v0y, v0z,
		&r1x, &r1y, &r1z, &v1x, &v1y, &v1z);									// ? FLOPs

	real_t jmu = jm / m;														// 1 FLOPs

	*idrx += jmu * (r1x - r0x);													// 3 FLOPs
	*idry += jmu * (r1y - r0y);													// 3 FLOPs
	*idrz += jmu * (r1z - r0z);													// 3 FLOPs
	*idvx += jmu * (v1x - v0x);													// 3 FLOPs
	*idvy += jmu * (v1y - v0y);													// 3 FLOPs
	*idvz += jmu * (v1z - v0z);													// 3 FLOPs
}
// Total flop count: 36 + ?


#endif	// __SAKURA_KERNEL_COMMON_H__
