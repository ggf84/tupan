#ifndef __SAKURA_KERNEL_COMMON_H__
#define __SAKURA_KERNEL_COMMON_H__

#include "common.h"
#include "universal_kepler_solver.h"


#define SAKURA_DECL_STRUCTS(iT, jT)			\
	typedef struct sakura_idata {			\
		iT drx, dry, drz, dvx, dvy, dvz;	\
		iT rx, ry, rz, vx, vy, vz, e2, m;	\
	} Sakura_IData;							\
	typedef struct sakura_jdata {			\
		jT rx, ry, rz, vx, vy, vz, e2, m;	\
	} Sakura_JData;

SAKURA_DECL_STRUCTS(real_t1, real_t)


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


static inline Sakura_IData
sakura_kernel_core(Sakura_IData ip, Sakura_JData jp,
				   real_t const dt, int_t const flag)
{
	real_t r0x = ip.rx - jp.rx;													// 1 FLOPs
	real_t r0y = ip.ry - jp.ry;													// 1 FLOPs
	real_t r0z = ip.rz - jp.rz;													// 1 FLOPs
	real_t v0x = ip.vx - jp.vx;													// 1 FLOPs
	real_t v0y = ip.vy - jp.vy;													// 1 FLOPs
	real_t v0z = ip.vz - jp.vz;													// 1 FLOPs
	real_t e2 = ip.e2 + jp.e2;													// 1 FLOPs
	real_t m = ip.m + jp.m;														// 1 FLOPs

	real_t r1x, r1y, r1z;
	real_t v1x, v1y, v1z;
	evolve_twobody(
		dt, flag, m, e2, r0x, r0y, r0z, v0x, v0y, v0z,
		&r1x, &r1y, &r1z, &v1x, &v1y, &v1z);									// ? FLOPs

	real_t jmu = jp.m / m;														// 1 FLOPs

	ip.drx += jmu * (r1x - r0x);												// 3 FLOPs
	ip.dry += jmu * (r1y - r0y);												// 3 FLOPs
	ip.drz += jmu * (r1z - r0z);												// 3 FLOPs
	ip.dvx += jmu * (v1x - v0x);												// 3 FLOPs
	ip.dvy += jmu * (v1y - v0y);												// 3 FLOPs
	ip.dvz += jmu * (v1z - v0z);												// 3 FLOPs
	return ip;
}
// Total flop count: 36 + ?


#endif	// __SAKURA_KERNEL_COMMON_H__
