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
		real_t m;
		real_t _m[1];
	};
	union {
		real_t e2;
		real_t _e2[1];
	};
	union {
		real_t rx;
		real_t _rx[1];
	};
	union {
		real_t ry;
		real_t _ry[1];
	};
	union {
		real_t rz;
		real_t _rz[1];
	};
	union {
		real_t vx;
		real_t _vx[1];
	};
	union {
		real_t vy;
		real_t _vy[1];
	};
	union {
		real_t vz;
		real_t _vz[1];
	};
	union {
		real_t drx;
		real_t _drx[1];
	};
	union {
		real_t dry;
		real_t _dry[1];
	};
	union {
		real_t drz;
		real_t _drz[1];
	};
	union {
		real_t dvx;
		real_t _dvx[1];
	};
	union {
		real_t dvy;
		real_t _dvy[1];
	};
	union {
		real_t dvz;
		real_t _dvz[1];
	};
} Sakura_Data;


typedef struct sakura_data_soa {
	union {
		real_t m[NLANES];
		real_t _m[NLANES * 1];
	};
	union {
		real_t e2[NLANES];
		real_t _e2[NLANES * 1];
	};
	union {
		real_t rx[NLANES];
		real_t _rx[NLANES * 1];
	};
	union {
		real_t ry[NLANES];
		real_t _ry[NLANES * 1];
	};
	union {
		real_t rz[NLANES];
		real_t _rz[NLANES * 1];
	};
	union {
		real_t vx[NLANES];
		real_t _vx[NLANES * 1];
	};
	union {
		real_t vy[NLANES];
		real_t _vy[NLANES * 1];
	};
	union {
		real_t vz[NLANES];
		real_t _vz[NLANES * 1];
	};
	union {
		real_t drx[NLANES];
		real_t _drx[NLANES * 1];
	};
	union {
		real_t dry[NLANES];
		real_t _dry[NLANES * 1];
	};
	union {
		real_t drz[NLANES];
		real_t _drz[NLANES * 1];
	};
	union {
		real_t dvx[NLANES];
		real_t _dvx[NLANES * 1];
	};
	union {
		real_t dvy[NLANES];
		real_t _dvy[NLANES * 1];
	};
	union {
		real_t dvz[NLANES];
		real_t _dvz[NLANES * 1];
	};
} Sakura_Data_SoA;


static inline void
read_Sakura_Data(
	uint_t base,
	uint_t lid,
	Sakura_Data *p,
	uint_t n,
	global const real_t __m[],
	global const real_t __e2[],
	global const real_t __rdot[])
{
	for (uint_t k = 0, kk = base + lid;
				k < 1;
				k += 1, kk += WGSIZE) {
		if (kk < n) {
			p->_m[k] = __m[kk];
			p->_e2[k] = __e2[kk];
			p->_rx[k] = (__rdot+(0*NDIM+0)*n)[kk];
			p->_ry[k] = (__rdot+(0*NDIM+1)*n)[kk];
			p->_rz[k] = (__rdot+(0*NDIM+2)*n)[kk];
			p->_vx[k] = (__rdot+(1*NDIM+0)*n)[kk];
			p->_vy[k] = (__rdot+(1*NDIM+1)*n)[kk];
			p->_vz[k] = (__rdot+(1*NDIM+2)*n)[kk];
		}
	}
}


static inline void
simd_shuff_Sakura_Data(Sakura_Data *p)
{
	shuff(p->m, 1);
	shuff(p->e2, 1);
	shuff(p->rx, 1);
	shuff(p->ry, 1);
	shuff(p->rz, 1);
	shuff(p->vx, 1);
	shuff(p->vy, 1);
	shuff(p->vz, 1);
	shuff(p->drx, 1);
	shuff(p->dry, 1);
	shuff(p->drz, 1);
	shuff(p->dvx, 1);
	shuff(p->dvy, 1);
	shuff(p->dvz, 1);
}


#endif	// __cplusplus
#endif	// __SAKURA_KERNEL_COMMON_H__
