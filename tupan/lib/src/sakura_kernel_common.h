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
				r1x[j] = r0x[j];
				r1y[j] = r0y[j];
				r1z[j] = r0z[j];
				v1x[j] = v0x[j];
				v1y[j] = v0y[j];
				v1z[j] = v0z[j];
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
		real_t m[LSIZE];
		real_t _m[LSIZE * 1];
	};
	union {
		real_t e2[LSIZE];
		real_t _e2[LSIZE * 1];
	};
	union {
		real_t rx[LSIZE];
		real_t _rx[LSIZE * 1];
	};
	union {
		real_t ry[LSIZE];
		real_t _ry[LSIZE * 1];
	};
	union {
		real_t rz[LSIZE];
		real_t _rz[LSIZE * 1];
	};
	union {
		real_t vx[LSIZE];
		real_t _vx[LSIZE * 1];
	};
	union {
		real_t vy[LSIZE];
		real_t _vy[LSIZE * 1];
	};
	union {
		real_t vz[LSIZE];
		real_t _vz[LSIZE * 1];
	};
	union {
		real_t drx[LSIZE];
		real_t _drx[LSIZE * 1];
	};
	union {
		real_t dry[LSIZE];
		real_t _dry[LSIZE * 1];
	};
	union {
		real_t drz[LSIZE];
		real_t _drz[LSIZE * 1];
	};
	union {
		real_t dvx[LSIZE];
		real_t _dvx[LSIZE * 1];
	};
	union {
		real_t dvy[LSIZE];
		real_t _dvy[LSIZE * 1];
	};
	union {
		real_t dvz[LSIZE];
		real_t _dvz[LSIZE * 1];
	};
} Sakura_Data;


static inline void
sakura_kernel_core(
	uint_t i, uint_t j,
	local Sakura_Data *ip,
	local Sakura_Data *jp,
	const real_t dt,
	const int_t flag)
// flop count: 27 + ??
{
	real_t m = ip->m[i] + jp->_m[j];
	real_t e2 = ip->e2[i] + jp->_e2[j];
	real_t r0x = ip->rx[i] - jp->_rx[j];
	real_t r0y = ip->ry[i] - jp->_ry[j];
	real_t r0z = ip->rz[i] - jp->_rz[j];
	real_t v0x = ip->vx[i] - jp->_vx[j];
	real_t v0y = ip->vy[i] - jp->_vy[j];
	real_t v0z = ip->vz[i] - jp->_vz[j];

	real_t r1x = r0x;
	real_t r1y = r0y;
	real_t r1z = r0z;
	real_t v1x = v0x;
	real_t v1y = v0y;
	real_t v1z = v0z;
	evolve_twobody(
		dt, flag, m, e2,
		r0x, r0y, r0z, v0x, v0y, v0z,
		&r1x, &r1y, &r1z, &v1x, &v1y, &v1z
	);	// flop count: ??

	real_t inv_m = 1 / m;
	real_t drx = r1x - r0x;
	real_t dry = r1y - r0y;
	real_t drz = r1z - r0z;
	real_t dvx = v1x - v0x;
	real_t dvy = v1y - v0y;
	real_t dvz = v1z - v0z;

	real_t jmu = jp->_m[j] * inv_m;

	ip->drx[i] += jmu * drx;
	ip->dry[i] += jmu * dry;
	ip->drz[i] += jmu * drz;
	ip->dvx[i] += jmu * dvx;
	ip->dvy[i] += jmu * dvy;
	ip->dvz[i] += jmu * dvz;
}


#endif	// __cplusplus
#endif	// __SAKURA_KERNEL_COMMON_H__
