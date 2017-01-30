#ifndef __TSTEP_KERNEL_COMMON_H__
#define __TSTEP_KERNEL_COMMON_H__


#include "common.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
namespace Tstep {

template<size_t TILE>
struct Tstep_Data_SoA {
	real_t m[TILE];
	real_t e2[TILE];
	real_t rx[TILE];
	real_t ry[TILE];
	real_t rz[TILE];
	real_t vx[TILE];
	real_t vy[TILE];
	real_t vz[TILE];
	real_t w2_a[TILE];
	real_t w2_b[TILE];
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
void commit(const uint_t n, const PART& part, real_t __dt_a[], real_t __dt_b[], const real_t eta)
{
	for (size_t k = 0; k < n; ++k) {
		auto kk = k%TILE;
		auto& p = part[k/TILE];
		__dt_a[k] = eta * rsqrt(p.w2_a[kk]);
		__dt_b[k] = eta * rsqrt(p.w2_b[kk]);
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
			auto im = ip.m[i];
			auto iee = ip.e2[i];
			auto irx = ip.rx[i];
			auto iry = ip.ry[i];
			auto irz = ip.rz[i];
			auto ivx = ip.vx[i];
			auto ivy = ip.vy[i];
			auto ivz = ip.vz[i];
			auto iw2_a = ip.w2_a[i];
			auto iw2_b = ip.w2_b[i];
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto m_r3 = im + jp.m[j];
				auto ee = iee + jp.e2[j];
				auto rx = irx - jp.rx[j];
				auto ry = iry - jp.ry[j];
				auto rz = irz - jp.rz[j];
				auto vx = ivx - jp.vx[j];
				auto vy = ivy - jp.vy[j];
				auto vz = ivz - jp.vz[j];

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

				m_r3 = (rr > ee && p.m[j] != 0) ? (m_r3):(0);

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
		real_tn m[LSIZE];
		real_t _m[LSIZE * SIMD];
	};
	union {
		real_tn e2[LSIZE];
		real_t _e2[LSIZE * SIMD];
	};
	union {
		real_tn rx[LSIZE];
		real_t _rx[LSIZE * SIMD];
	};
	union {
		real_tn ry[LSIZE];
		real_t _ry[LSIZE * SIMD];
	};
	union {
		real_tn rz[LSIZE];
		real_t _rz[LSIZE * SIMD];
	};
	union {
		real_tn vx[LSIZE];
		real_t _vx[LSIZE * SIMD];
	};
	union {
		real_tn vy[LSIZE];
		real_t _vy[LSIZE * SIMD];
	};
	union {
		real_tn vz[LSIZE];
		real_t _vz[LSIZE * SIMD];
	};
	union {
		real_tn w2_a[LSIZE];
		real_t _w2_a[LSIZE * SIMD];
	};
	union {
		real_tn w2_b[LSIZE];
		real_t _w2_b[LSIZE * SIMD];
	};
} Tstep_Data;


#endif	// __cplusplus
#endif	// __TSTEP_KERNEL_COMMON_H__
