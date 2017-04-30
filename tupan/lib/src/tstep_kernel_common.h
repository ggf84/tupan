#ifndef __TSTEP_KERNEL_COMMON_H__
#define __TSTEP_KERNEL_COMMON_H__


#include "common.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
namespace Tstep {

template<size_t TILE>
struct Tstep_Data_SoA {
	real_t __ALIGNED__ m[TILE];
	real_t __ALIGNED__ e2[TILE];
	real_t __ALIGNED__ rx[TILE];
	real_t __ALIGNED__ ry[TILE];
	real_t __ALIGNED__ rz[TILE];
	real_t __ALIGNED__ vx[TILE];
	real_t __ALIGNED__ vy[TILE];
	real_t __ALIGNED__ vz[TILE];
	real_t __ALIGNED__ w2_a[TILE];
	real_t __ALIGNED__ w2_b[TILE];
};

template<size_t TILE>
auto setup(
	const uint_t n,
	const real_t __m[],
	const real_t __e2[],
	const real_t __rdot[])
{
	auto ntiles = (n + TILE - 1) / TILE;
	vector<Tstep_Data_SoA<TILE>> part(ntiles);
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
void commit(const uint_t n, const PART& part, real_t __w2_a[], real_t __w2_b[], const real_t eta)
{
	for (size_t k = 0; k < n; ++k) {
		auto kk = k%TILE;
		auto& p = part[k/TILE];
		__w2_a[k] = p.w2_a[kk];
		__w2_b[k] = p.w2_b[kk];
	}
}

template<size_t TILE>
struct P2P_tstep_kernel_core {
	const real_t eta;
	P2P_tstep_kernel_core(const real_t& eta) : eta(eta) {}

	template<typename IP, typename JP>
	void operator()(IP&& ip, JP&& jp) {
		// flop count: 43
		for (size_t i = 0; i < TILE; ++i) {
			auto iw2_a = ip.w2_a[i];
			auto iw2_b = ip.w2_b[i];
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto m_r3 = ip.m[i] + jp.m[j];
				auto ee = ip.e2[i] + jp.e2[j];
				auto rx = ip.rx[i] - jp.rx[j];
				auto ry = ip.ry[i] - jp.ry[j];
				auto rz = ip.rz[i] - jp.rz[j];
				auto vx = ip.vx[i] - jp.vx[j];
				auto vy = ip.vy[i] - jp.vy[j];
				auto vz = ip.vz[i] - jp.vz[j];

				auto rr = ee;
				rr     += rx * rx + ry * ry + rz * rz;
				auto rv = rx * vx + ry * vy + rz * vz;
				auto vv = vx * vx + vy * vy + vz * vz;

				auto inv_r2 = rsqrt(rr);
				m_r3 *= inv_r2;
				inv_r2 *= inv_r2;
				m_r3 *= 2 * inv_r2;

				auto m_r5 = m_r3 * inv_r2;
				m_r3 += vv * inv_r2;
				rv *= eta * rsqrt(m_r3);
				m_r5 += m_r3 * inv_r2;
				m_r3 -= m_r5 * rv;

				m_r3 = (ip.m[i] == 0 || jp.m[j] == 0) ? (0):(m_r3);

				jp.w2_a[j] = fmax(m_r3, jp.w2_a[j]);
				jp.w2_b[j] += m_r3;

				iw2_a = fmax(m_r3, iw2_a);
				iw2_b += m_r3;
			}
			ip.w2_a[i] = iw2_a;
			ip.w2_b[i] = iw2_b;
		}
	}

	template<typename P>
	void operator()(P&& p) {
		// flop count: 42
		for (size_t i = 0; i < TILE; ++i) {
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto m_r3 = p.m[i] + p.m[j];
				auto ee = p.e2[i] + p.e2[j];
				auto rx = p.rx[i] - p.rx[j];
				auto ry = p.ry[i] - p.ry[j];
				auto rz = p.rz[i] - p.rz[j];
				auto vx = p.vx[i] - p.vx[j];
				auto vy = p.vy[i] - p.vy[j];
				auto vz = p.vz[i] - p.vz[j];

				auto rr = ee;
				rr     += rx * rx + ry * ry + rz * rz;
				auto rv = rx * vx + ry * vy + rz * vz;
				auto vv = vx * vx + vy * vy + vz * vz;

				auto inv_r2 = rsqrt(rr);
				m_r3 *= inv_r2;
				inv_r2 *= inv_r2;
				m_r3 *= 2 * inv_r2;

				auto m_r5 = m_r3 * inv_r2;
				m_r3 += vv * inv_r2;
				rv *= eta * rsqrt(m_r3);
				m_r5 += m_r3 * inv_r2;
				m_r3 -= m_r5 * rv;

				m_r3 = (rr > ee) ? (m_r3):(0);
				m_r3 = (p.m[i] == 0 || p.m[j] == 0) ? (0):(m_r3);

				p.w2_a[j] = fmax(m_r3, p.w2_a[j]);
				p.w2_b[j] += m_r3;
			}
		}
	}
};

}	// namespace Tstep
#else


// ----------------------------------------------------------------------------


typedef struct tstep_data {
	union {
		real_tn m[WPT];
		real_t _m[WPT * SIMD];
	};
	union {
		real_tn e2[WPT];
		real_t _e2[WPT * SIMD];
	};
	union {
		real_tn rx[WPT];
		real_t _rx[WPT * SIMD];
	};
	union {
		real_tn ry[WPT];
		real_t _ry[WPT * SIMD];
	};
	union {
		real_tn rz[WPT];
		real_t _rz[WPT * SIMD];
	};
	union {
		real_tn vx[WPT];
		real_t _vx[WPT * SIMD];
	};
	union {
		real_tn vy[WPT];
		real_t _vy[WPT * SIMD];
	};
	union {
		real_tn vz[WPT];
		real_t _vz[WPT * SIMD];
	};
	union {
		real_tn w2_a[WPT];
		real_t _w2_a[WPT * SIMD];
	};
	union {
		real_tn w2_b[WPT];
		real_t _w2_b[WPT * SIMD];
	};
} Tstep_Data;


typedef struct tstep_data_soa {
	union {
		real_tn m[NLANES];
		real_t _m[NLANES * SIMD];
	};
	union {
		real_tn e2[NLANES];
		real_t _e2[NLANES * SIMD];
	};
	union {
		real_tn rx[NLANES];
		real_t _rx[NLANES * SIMD];
	};
	union {
		real_tn ry[NLANES];
		real_t _ry[NLANES * SIMD];
	};
	union {
		real_tn rz[NLANES];
		real_t _rz[NLANES * SIMD];
	};
	union {
		real_tn vx[NLANES];
		real_t _vx[NLANES * SIMD];
	};
	union {
		real_tn vy[NLANES];
		real_t _vy[NLANES * SIMD];
	};
	union {
		real_tn vz[NLANES];
		real_t _vz[NLANES * SIMD];
	};
	union {
		real_tn w2_a[NLANES];
		real_t _w2_a[NLANES * SIMD];
	};
	union {
		real_tn w2_b[NLANES];
		real_t _w2_b[NLANES * SIMD];
	};
} Tstep_Data_SoA;


static inline void
read_Tstep_Data(
	Tstep_Data *p,
	const uint_t base,
	const uint_t stride,
	const uint_t nloads,
	const uint_t n,
	global const real_t __m[],
	global const real_t __e2[],
	global const real_t __rdot[])
{
	for (uint_t k = 0, kk = base;
				k < nloads;
				k += 1, kk += stride) {
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
simd_shuff_Tstep_Data(
	const uint_t k,
	Tstep_Data *p)
{
	shuff(p->m[k], SIMD);
	shuff(p->e2[k], SIMD);
	shuff(p->rx[k], SIMD);
	shuff(p->ry[k], SIMD);
	shuff(p->rz[k], SIMD);
	shuff(p->vx[k], SIMD);
	shuff(p->vy[k], SIMD);
	shuff(p->vz[k], SIMD);
	shuff(p->w2_a[k], SIMD);
	shuff(p->w2_b[k], SIMD);
}


#endif	// __cplusplus
#endif	// __TSTEP_KERNEL_COMMON_H__
