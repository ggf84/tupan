#ifndef __SAKURA_KERNEL_COMMON_H__
#define __SAKURA_KERNEL_COMMON_H__

#include "common.h"
#include "universal_kepler_solver.h"


#define SAKURA_IMPLEMENT_STRUCT(N)				\
	typedef struct concat(sakura_data, N) {		\
		concat(real_t, N) m, e2;				\
		concat(real_t, N) rdot[2][NDIM];		\
		concat(real_t, N) drdot[2][NDIM];		\
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
	real_t r0dot[2][NDIM];
	#pragma unroll
	for (uint_t kdot = 0; kdot < 2; ++kdot) {
		#pragma unroll
		for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
			r0dot[kdot][kdim] = ip.rdot[kdot][kdim] - jp.rdot[kdot][kdim];
		}
	}
	real_t e2 = ip.e2 + jp.e2;
	real_t m = ip.m + jp.m;

	real_t r1dot[2][NDIM];
	evolve_twobody(
		dt, flag, m, e2,
		r0dot[0][0], r0dot[0][1], r0dot[0][2],
		r0dot[1][0], r0dot[1][1], r0dot[1][2],
		&r1dot[0][0], &r1dot[0][1], &r1dot[0][2],
		&r1dot[1][0], &r1dot[1][1], &r1dot[1][2]
	);	// flop count: ??

	real_t drdot[2][NDIM];
	#pragma unroll
	for (uint_t kdot = 0; kdot < 2; ++kdot) {
		#pragma unroll
		for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
			drdot[kdot][kdim] = r1dot[kdot][kdim] - r0dot[kdot][kdim];
		}
	}

	real_t inv_m = 1 / m;
	real_t jmu = jp.m * inv_m;
	#pragma unroll
	for (uint_t kdot = 0; kdot < 2; ++kdot) {
		#pragma unroll
		for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
			ip.drdot[kdot][kdim] += jmu * drdot[kdot][kdim];
		}
	}
	return ip;
}

// ----------------------------------------------------------------------------

#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
template<typename I, typename J, typename PARAM, typename FLAG>
static inline void
p2p_sakura_kernel_core(I &ip, J &jp, const PARAM dt, const FLAG flag)
// flop count: 41 + ??
{
	decltype(ip.rdot) r0dot;
	for (auto kdot = 0; kdot < 2; ++kdot) {
		for (auto kdim = 0; kdim < NDIM; ++kdim) {
			r0dot[kdot][kdim] = ip.rdot[kdot][kdim] - jp.rdot[kdot][kdim];
		}
	}
	auto e2 = ip.e2 + jp.e2;
	auto m = ip.m + jp.m;

	decltype(ip.rdot) r1dot;
	evolve_twobody(
		dt, flag, m, e2,
		r0dot[0][0], r0dot[0][1], r0dot[0][2],
		r0dot[1][0], r0dot[1][1], r0dot[1][2],
		&r1dot[0][0], &r1dot[0][1], &r1dot[0][2],
		&r1dot[1][0], &r1dot[1][1], &r1dot[1][2]
	);	// flop count: ??

	decltype(ip.rdot) drdot;
	for (auto kdot = 0; kdot < 2; ++kdot) {
		for (auto kdim = 0; kdim < NDIM; ++kdim) {
			drdot[kdot][kdim] = r1dot[kdot][kdim] - r0dot[kdot][kdim];
		}
	}

	auto inv_m = 1 / m;
	{	// i-particle
		auto jmu = jp.m * inv_m;
		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				ip.drdot[kdot][kdim] += jmu * drdot[kdot][kdim];
			}
		}
	}
	{	// j-particle
		auto imu = ip.m * inv_m;
		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				jp.drdot[kdot][kdim] -= imu * drdot[kdot][kdim];
			}
		}
	}
}
#endif

#endif	// __SAKURA_KERNEL_COMMON_H__
