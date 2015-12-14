#ifndef __SAKURA_KERNEL_COMMON_H__
#define __SAKURA_KERNEL_COMMON_H__

#include "common.h"
#include "universal_kepler_solver.h"


#define SAKURA_IMPLEMENT_STRUCT(N)							\
	typedef struct concat(sakura_data, N) {					\
		concat(real_t, N) drx, dry, drz, dvx, dvy, dvz;		\
		concat(real_t, N) rx, ry, rz, vx, vy, vz, e2, m;	\
	} concat(Sakura_Data, N);

SAKURA_IMPLEMENT_STRUCT(1)
#if SIMD > 1
SAKURA_IMPLEMENT_STRUCT(SIMD)
#endif
typedef Sakura_Data1 Sakura_Data;


static inline void
twobody_solver(
	const real_t dt,
	const real_t m,
	const real_t e2,
	const real_t r0x,
	const real_t r0y,
	const real_t r0z,
	const real_t v0x,
	const real_t v0y,
	const real_t v0z,
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
	const real_t dt,
	const int_t flag,
	const real_t m,
	const real_t e2,
	const real_t r0x,
	const real_t r0y,
	const real_t r0z,
	const real_t v0x,
	const real_t v0y,
	const real_t v0z,
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
		rx -= vx * dt;
		ry -= vy * dt;
		rz -= vz * dt;
		twobody_solver(
			dt, m, e2, rx, ry, rz, vx, vy, vz,
			&rx, &ry, &rz, &vx, &vy, &vz);	// flop count: ??
	}
	if (flag == 1) {
		twobody_solver(
			dt, m, e2, rx, ry, rz, vx, vy, vz,
			&rx, &ry, &rz, &vx, &vy, &vz);	// flop count: ??
		rx -= vx * dt;
		ry -= vy * dt;
		rz -= vz * dt;
	}
	if (flag == -2) {
		rx -= vx * dt / 2;
		ry -= vy * dt / 2;
		rz -= vz * dt / 2;
		twobody_solver(
			dt, m, e2, rx, ry, rz, vx, vy, vz,
			&rx, &ry, &rz, &vx, &vy, &vz);	// flop count: ??
		rx -= vx * dt / 2;
		ry -= vy * dt / 2;
		rz -= vz * dt / 2;
	}
	if (flag == 2) {
		twobody_solver(
			dt/2, m, e2, rx, ry, rz, vx, vy, vz,
			&rx, &ry, &rz, &vx, &vy, &vz);	// flop count: ??
		rx -= vx * dt;
		ry -= vy * dt;
		rz -= vz * dt;
		twobody_solver(
			dt/2, m, e2, rx, ry, rz, vx, vy, vz,
			&rx, &ry, &rz, &vx, &vy, &vz);	// flop count: ??
	}

	*r1x = rx;
	*r1y = ry;
	*r1z = rz;
	*v1x = vx;
	*v1y = vy;
	*v1z = vz;
}


static inline Sakura_Data1
sakura_kernel_core(Sakura_Data1 ip, Sakura_Data jp,
				   const real_t dt, const int_t flag)
// flop count: 27 + ??
{
	real_t r0x = ip.rx - jp.rx;
	real_t r0y = ip.ry - jp.ry;
	real_t r0z = ip.rz - jp.rz;
	real_t v0x = ip.vx - jp.vx;
	real_t v0y = ip.vy - jp.vy;
	real_t v0z = ip.vz - jp.vz;
	real_t e2 = ip.e2 + jp.e2;
	real_t m = ip.m + jp.m;

	real_t r1x, r1y, r1z;
	real_t v1x, v1y, v1z;
	evolve_twobody(
		dt, flag, m, e2, r0x, r0y, r0z, v0x, v0y, v0z,
		&r1x, &r1y, &r1z, &v1x, &v1y, &v1z);	// flop count: ??

	real_t jmu = jp.m / m;

	ip.drx += jmu * (r1x - r0x);
	ip.dry += jmu * (r1y - r0y);
	ip.drz += jmu * (r1z - r0z);
	ip.dvx += jmu * (v1x - v0x);
	ip.dvy += jmu * (v1y - v0y);
	ip.dvz += jmu * (v1z - v0z);
	return ip;
}


#endif	// __SAKURA_KERNEL_COMMON_H__
