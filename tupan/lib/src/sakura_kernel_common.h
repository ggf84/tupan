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
sakura_kernel_core(Sakura_Data1 ip, Sakura_Data1 jp,
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
template<size_t TILE>
struct Sakura_Data_SoA {
	real_t m[TILE];
	real_t e2[TILE];
	real_t rx[TILE];
	real_t ry[TILE];
	real_t rz[TILE];
	real_t vx[TILE];
	real_t vy[TILE];
	real_t vz[TILE];
	real_t drx[TILE];
	real_t dry[TILE];
	real_t drz[TILE];
	real_t dvx[TILE];
	real_t dvy[TILE];
	real_t dvz[TILE];
};

template<size_t TILE>
struct P2P_sakura_kernel_core {
	const real_t dt;
	const int_t flag;
	P2P_sakura_kernel_core(const real_t& dt, const int_t& flag) :
		dt(dt), flag(flag)
		{}

	template<typename IP, typename JP>
	void operator()(IP&& ip, JP&& jp) {
		// flop count: 41 + ???
		for (size_t i = 0; i < TILE; ++i) {
			#pragma unroll
			for (size_t j = 0; j < TILE; ++j) {
				auto m = ip.m[i] + jp.m[j];
				auto e2 = ip.e2[i] + jp.e2[j];
				auto r0x = ip.rx[i] - jp.rx[j];
				auto r0y = ip.ry[i] - jp.ry[j];
				auto r0z = ip.rz[i] - jp.rz[j];
				auto v0x = ip.vx[i] - jp.vx[j];
				auto v0y = ip.vy[i] - jp.vy[j];
				auto v0z = ip.vz[i] - jp.vz[j];

				decltype(r0x) r1x, r1y, r1z;
				decltype(v0x) v1x, v1y, v1z;
				evolve_twobody(
					dt, flag, m, e2,
					r0x, r0y, r0z, v0x, v0y, v0z,
					&r1x, &r1y, &r1z, &v1x, &v1y, &v1z
				);	// flop count: ??

				auto inv_m = 1 / m;
				auto drx = r1x - r0x;
				auto dry = r1y - r0y;
				auto drz = r1z - r0z;
				auto dvx = v1x - v0x;
				auto dvy = v1y - v0y;
				auto dvz = v1z - v0z;

				{	// i-particle
					auto jmu = jp.m[j] * inv_m;
					ip.drx[i] += jmu * drx;
					ip.dry[i] += jmu * dry;
					ip.drz[i] += jmu * drz;
					ip.dvx[i] += jmu * dvx;
					ip.dvy[i] += jmu * dvy;
					ip.dvz[i] += jmu * dvz;
				}
				{	// j-particle
					auto imu = ip.m[i] * inv_m;
					jp.drx[j] -= imu * drx;
					jp.dry[j] -= imu * dry;
					jp.drz[j] -= imu * drz;
					jp.dvx[j] -= imu * dvx;
					jp.dvy[j] -= imu * dvy;
					jp.dvz[j] -= imu * dvz;
				}
			}
		}
	}

	template<typename P>
	void operator()(P&& p) {
		// flop count: 28 + ???
		for (size_t i = 0; i < TILE; ++i) {
			#pragma unroll
			for (size_t j = 0; j < TILE; ++j) {
				if (i == j) continue;
				auto m = p.m[i] + p.m[j];
				auto e2 = p.e2[i] + p.e2[j];
				auto r0x = p.rx[i] - p.rx[j];
				auto r0y = p.ry[i] - p.ry[j];
				auto r0z = p.rz[i] - p.rz[j];
				auto v0x = p.vx[i] - p.vx[j];
				auto v0y = p.vy[i] - p.vy[j];
				auto v0z = p.vz[i] - p.vz[j];

				decltype(r0x) r1x, r1y, r1z;
				decltype(v0x) v1x, v1y, v1z;
				evolve_twobody(
					dt, flag, m, e2,
					r0x, r0y, r0z, v0x, v0y, v0z,
					&r1x, &r1y, &r1z, &v1x, &v1y, &v1z
				);	// flop count: ??

				auto inv_m = 1 / m;
				auto drx = r1x - r0x;
				auto dry = r1y - r0y;
				auto drz = r1z - r0z;
				auto dvx = v1x - v0x;
				auto dvy = v1y - v0y;
				auto dvz = v1z - v0z;

				auto jmu = p.m[j] * inv_m;
				p.drx[i] += jmu * drx;
				p.dry[i] += jmu * dry;
				p.drz[i] += jmu * drz;
				p.dvx[i] += jmu * dvx;
				p.dvy[i] += jmu * dvy;
				p.dvz[i] += jmu * dvz;
			}
		}
	}
};
#endif

#endif	// __SAKURA_KERNEL_COMMON_H__
