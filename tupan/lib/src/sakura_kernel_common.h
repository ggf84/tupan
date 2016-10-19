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

template<size_t TILE, typename T = Sakura_Data_SoA<TILE>>
auto setup(
	const uint_t n,
	const real_t __m[],
	const real_t __e2[],
	const real_t __rdot[])
{
	auto ntiles = (n + TILE - 1) / TILE;
	vector<T> part(ntiles);
	for (size_t k = 0; k < n; ++k) {
		auto kk = k%TILE;
		auto& p = part[k/TILE];
		p.m[kk] = __m[k];
		p.e2[kk] = __e2[k];
		p.rx[kk] = __rdot[(0*NDIM+0)*n + k];
		p.ry[kk] = __rdot[(0*NDIM+1)*n + k];
		p.rz[kk] = __rdot[(0*NDIM+2)*n + k];
		p.vx[kk] = __rdot[(1*NDIM+0)*n + k];
		p.vy[kk] = __rdot[(1*NDIM+1)*n + k];
		p.vz[kk] = __rdot[(1*NDIM+2)*n + k];
	}
	return part;
}

template<size_t TILE, typename PART>
void commit(const uint_t n, const PART& part, real_t __drdot[])
{
	for (size_t k = 0; k < n; ++k) {
		auto kk = k%TILE;
		auto& p = part[k/TILE];
		__drdot[(0*NDIM+0)*n + k] = p.drx[kk];
		__drdot[(0*NDIM+1)*n + k] = p.dry[kk];
		__drdot[(0*NDIM+2)*n + k] = p.drz[kk];
		__drdot[(1*NDIM+0)*n + k] = p.dvx[kk];
		__drdot[(1*NDIM+1)*n + k] = p.dvy[kk];
		__drdot[(1*NDIM+2)*n + k] = p.dvz[kk];
	}
}

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
		decltype(jp.m) m, e2, imu, jmu;
		decltype(jp.m) r0x, r0y, r0z, v0x, v0y, v0z;
		decltype(jp.m) r1x, r1y, r1z, v1x, v1y, v1z;
		decltype(jp.m) drx, dry, drz, dvx, dvy, dvz;
		for (size_t i = 0; i < TILE; ++i) {
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				m[j] = ip.m[i] + jp.m[j];
				e2[j] = ip.e2[i] + jp.e2[j];
				r0x[j] = ip.rx[i] - jp.rx[j];
				r0y[j] = ip.ry[i] - jp.ry[j];
				r0z[j] = ip.rz[i] - jp.rz[j];
				v0x[j] = ip.vx[i] - jp.vx[j];
				v0y[j] = ip.vy[i] - jp.vy[j];
				v0z[j] = ip.vz[i] - jp.vz[j];
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				evolve_twobody(
					dt, flag, m[j], e2[j],
					r0x[j], r0y[j], r0z[j], v0x[j], v0y[j], v0z[j],
					&r1x[j], &r1y[j], &r1z[j], &v1x[j], &v1y[j], &v1z[j]
				);	// flop count: ??
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto inv_m = 1 / m[j];
				drx[j] = r1x[j] - r0x[j];
				dry[j] = r1y[j] - r0y[j];
				drz[j] = r1z[j] - r0z[j];
				dvx[j] = v1x[j] - v0x[j];
				dvy[j] = v1y[j] - v0y[j];
				dvz[j] = v1z[j] - v0z[j];

				imu[j] = ip.m[i] * inv_m;
				jmu[j] = jp.m[j] * inv_m;
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				jp.drx[j] -= imu[j] * drx[j];
				jp.dry[j] -= imu[j] * dry[j];
				jp.drz[j] -= imu[j] * drz[j];
				jp.dvx[j] -= imu[j] * dvx[j];
				jp.dvy[j] -= imu[j] * dvy[j];
				jp.dvz[j] -= imu[j] * dvz[j];
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				ip.drx[i] += jmu[j] * drx[j];
				ip.dry[i] += jmu[j] * dry[j];
				ip.drz[i] += jmu[j] * drz[j];
				ip.dvx[i] += jmu[j] * dvx[j];
				ip.dvy[i] += jmu[j] * dvy[j];
				ip.dvz[i] += jmu[j] * dvz[j];
			}
		}
	}

	template<typename P>
	void operator()(P&& p) {
		// flop count: 28 + ???
		decltype(p.m) m, e2, imu;
		decltype(p.m) r0x, r0y, r0z, v0x, v0y, v0z;
		decltype(p.m) r1x, r1y, r1z, v1x, v1y, v1z;
		decltype(p.m) drx, dry, drz, dvx, dvy, dvz;
		for (size_t i = 0; i < TILE; ++i) {
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				m[j] = p.m[i] + p.m[j];
				e2[j] = p.e2[i] + p.e2[j];
				r0x[j] = p.rx[i] - p.rx[j];
				r0y[j] = p.ry[i] - p.ry[j];
				r0z[j] = p.rz[i] - p.rz[j];
				v0x[j] = p.vx[i] - p.vx[j];
				v0y[j] = p.vy[i] - p.vy[j];
				v0z[j] = p.vz[i] - p.vz[j];
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				if (i == j) {
					r1x[j] = r0x[j];
					r1y[j] = r0y[j];
					r1z[j] = r0z[j];
					v1x[j] = v0x[j];
					v1y[j] = v0y[j];
					v1z[j] = v0z[j];
				} else {
					evolve_twobody(
						dt, flag, m[j], e2[j],
						r0x[j], r0y[j], r0z[j], v0x[j], v0y[j], v0z[j],
						&r1x[j], &r1y[j], &r1z[j], &v1x[j], &v1y[j], &v1z[j]
					);	// flop count: ??
				}
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto inv_m = 1 / m[j];
				drx[j] = r1x[j] - r0x[j];
				dry[j] = r1y[j] - r0y[j];
				drz[j] = r1z[j] - r0z[j];
				dvx[j] = v1x[j] - v0x[j];
				dvy[j] = v1y[j] - v0y[j];
				dvz[j] = v1z[j] - v0z[j];

				imu[j] = p.m[i] * inv_m;
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				p.drx[j] -= imu[j] * drx[j];
				p.dry[j] -= imu[j] * dry[j];
				p.drz[j] -= imu[j] * drz[j];
				p.dvx[j] -= imu[j] * dvx[j];
				p.dvy[j] -= imu[j] * dvy[j];
				p.dvz[j] -= imu[j] * dvz[j];
			}
		}
	}
};
#endif

#endif	// __SAKURA_KERNEL_COMMON_H__
