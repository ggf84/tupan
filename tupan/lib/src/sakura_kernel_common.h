#ifndef __SAKURA_KERNEL_COMMON_H__
#define __SAKURA_KERNEL_COMMON_H__

#include "common.h"
#include "universal_kepler_solver.h"


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


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
namespace Sakura {

template<size_t TILE>
struct Sakura_Data_SoA {
	real_t __ALIGNED__ m[TILE];
	real_t __ALIGNED__ e2[TILE];
	real_t __ALIGNED__ rx[TILE];
	real_t __ALIGNED__ ry[TILE];
	real_t __ALIGNED__ rz[TILE];
	real_t __ALIGNED__ vx[TILE];
	real_t __ALIGNED__ vy[TILE];
	real_t __ALIGNED__ vz[TILE];
	real_t __ALIGNED__ drx[TILE];
	real_t __ALIGNED__ dry[TILE];
	real_t __ALIGNED__ drz[TILE];
	real_t __ALIGNED__ dvx[TILE];
	real_t __ALIGNED__ dvy[TILE];
	real_t __ALIGNED__ dvz[TILE];
};

template<size_t TILE>
auto setup(
	const uint_t n,
	const real_t __m[],
	const real_t __e2[],
	const real_t __rdot[])
{
	auto ntiles = (n + TILE - 1) / TILE;
	vector<Sakura_Data_SoA<TILE>> part(ntiles);
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
		for (size_t i = 0; i < TILE; ++i) {
			auto idrx = ip.drx[i];
			auto idry = ip.dry[i];
			auto idrz = ip.drz[i];
			auto idvx = ip.dvx[i];
			auto idvy = ip.dvy[i];
			auto idvz = ip.dvz[i];
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto m = ip.m[i] + jp.m[j];
				auto ee = ip.e2[i] + jp.e2[j];
				auto r0x = ip.rx[i] - jp.rx[j];
				auto r0y = ip.ry[i] - jp.ry[j];
				auto r0z = ip.rz[i] - jp.rz[j];
				auto v0x = ip.vx[i] - jp.vx[j];
				auto v0y = ip.vy[i] - jp.vy[j];
				auto v0z = ip.vz[i] - jp.vz[j];

				auto r1x = r0x;
				auto r1y = r0y;
				auto r1z = r0z;
				auto v1x = v0x;
				auto v1y = v0y;
				auto v1z = v0z;
				evolve_twobody(
					dt, flag, m, ee,
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

				auto imu = ip.m[i] * inv_m;
				jp.drx[j] -= imu * drx;
				jp.dry[j] -= imu * dry;
				jp.drz[j] -= imu * drz;
				jp.dvx[j] -= imu * dvx;
				jp.dvy[j] -= imu * dvy;
				jp.dvz[j] -= imu * dvz;

				auto jmu = jp.m[j] * inv_m;
				idrx += jmu * drx;
				idry += jmu * dry;
				idrz += jmu * drz;
				idvx += jmu * dvx;
				idvy += jmu * dvy;
				idvz += jmu * dvz;
			}
			ip.drx[i] = idrx;
			ip.dry[i] = idry;
			ip.drz[i] = idrz;
			ip.dvx[i] = idvx;
			ip.dvy[i] = idvy;
			ip.dvz[i] = idvz;
		}
	}

	template<typename P>
	void operator()(P&& p) {
		// flop count: 28 + ???
		for (size_t i = 0; i < TILE; ++i) {
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto m = p.m[i] + p.m[j];
				auto e2 = p.e2[i] + p.e2[j];
				auto r0x = p.rx[i] - p.rx[j];
				auto r0y = p.ry[i] - p.ry[j];
				auto r0z = p.rz[i] - p.rz[j];
				auto v0x = p.vx[i] - p.vx[j];
				auto v0y = p.vy[i] - p.vy[j];
				auto v0z = p.vz[i] - p.vz[j];

				auto r1x = r0x;
				auto r1y = r0y;
				auto r1z = r0z;
				auto v1x = v0x;
				auto v1y = v0y;
				auto v1z = v0z;
				if (i != j) {
					evolve_twobody(
						dt, flag, m, e2,
						r0x, r0y, r0z, v0x, v0y, v0z,
						&r1x, &r1y, &r1z, &v1x, &v1y, &v1z
					);	// flop count: ??
				}

				auto inv_m = 1 / m;
				auto drx = r1x - r0x;
				auto dry = r1y - r0y;
				auto drz = r1z - r0z;
				auto dvx = v1x - v0x;
				auto dvy = v1y - v0y;
				auto dvz = v1z - v0z;

				auto imu = p.m[i] * inv_m;

				p.drx[j] -= imu * drx;
				p.dry[j] -= imu * dry;
				p.drz[j] -= imu * drz;
				p.dvx[j] -= imu * dvx;
				p.dvy[j] -= imu * dvy;
				p.dvz[j] -= imu * dvz;
			}
		}
	}
};

}	// namespace Sakura
#else


// ----------------------------------------------------------------------------


typedef struct sakura_data {
	union {
		real_t m[LMSIZE];
		real_t _m[LMSIZE * 1];
	};
	union {
		real_t e2[LMSIZE];
		real_t _e2[LMSIZE * 1];
	};
	union {
		real_t rx[LMSIZE];
		real_t _rx[LMSIZE * 1];
	};
	union {
		real_t ry[LMSIZE];
		real_t _ry[LMSIZE * 1];
	};
	union {
		real_t rz[LMSIZE];
		real_t _rz[LMSIZE * 1];
	};
	union {
		real_t vx[LMSIZE];
		real_t _vx[LMSIZE * 1];
	};
	union {
		real_t vy[LMSIZE];
		real_t _vy[LMSIZE * 1];
	};
	union {
		real_t vz[LMSIZE];
		real_t _vz[LMSIZE * 1];
	};
	union {
		real_t drx[LMSIZE];
		real_t _drx[LMSIZE * 1];
	};
	union {
		real_t dry[LMSIZE];
		real_t _dry[LMSIZE * 1];
	};
	union {
		real_t drz[LMSIZE];
		real_t _drz[LMSIZE * 1];
	};
	union {
		real_t dvx[LMSIZE];
		real_t _dvx[LMSIZE * 1];
	};
	union {
		real_t dvy[LMSIZE];
		real_t _dvy[LMSIZE * 1];
	};
	union {
		real_t dvz[LMSIZE];
		real_t _dvz[LMSIZE * 1];
	};
} Sakura_Data;


#endif	// __cplusplus
#endif	// __SAKURA_KERNEL_COMMON_H__
