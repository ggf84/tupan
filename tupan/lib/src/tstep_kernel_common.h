#ifndef __TSTEP_KERNEL_COMMON_H__
#define __TSTEP_KERNEL_COMMON_H__


#include "common.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
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

template<size_t TILE, typename T = Tstep_Data_SoA<TILE>>
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
		decltype(jp.m) rv, vv, m_r3, inv_r2;
		for (size_t i = 0; i < TILE; ++i) {
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				m_r3[j] = ip.m[i] + jp.m[j];
				auto ee = ip.e2[i] + jp.e2[j];
				auto rx = ip.rx[i] - jp.rx[j];
				auto ry = ip.ry[i] - jp.ry[j];
				auto rz = ip.rz[i] - jp.rz[j];
				auto vx = ip.vx[i] - jp.vx[j];
				auto vy = ip.vy[i] - jp.vy[j];
				auto vz = ip.vz[i] - jp.vz[j];

				auto rr = ee;
				rr   += rx * rx + ry * ry + rz * rz;
				rv[j] = rx * vx + ry * vy + rz * vz;
				vv[j] = vx * vx + vy * vy + vz * vz;

				inv_r2[j] = rsqrt(rr);
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				m_r3[j] *= inv_r2[j];
				inv_r2[j] *= inv_r2[j];
				m_r3[j] *= 2 * inv_r2[j];

				auto m_r5 = m_r3[j] * inv_r2[j];
				m_r3[j] += vv[j] * inv_r2[j];
				rv[j] *= eta * rsqrt(m_r3[j]);
				m_r5 += m_r3[j] * inv_r2[j];
				m_r3[j] -= m_r5 * rv[j];
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				jp.w2_a[j] = fmax(m_r3[j], jp.w2_a[j]);
				jp.w2_b[j] += m_r3[j];
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				ip.w2_a[i] = fmax(m_r3[j], ip.w2_a[i]);
				ip.w2_b[i] += m_r3[j];
			}
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

				p.w2_a[j] = fmax(m_r3, p.w2_a[j]);
				p.w2_b[j] += m_r3;
			}
		}
	}
};
#endif


// ----------------------------------------------------------------------------


typedef struct tstep_data {
	real_tn m;
	real_tn e2;
	real_tn rx;
	real_tn ry;
	real_tn rz;
	real_tn vx;
	real_tn vy;
	real_tn vz;
	real_tn w2_a;
	real_tn w2_b;
} Tstep_Data;


static inline Tstep_Data
tstep_kernel_core(Tstep_Data ip, Tstep_Data jp, const real_t eta)
// flop count: 42
{
	real_tn m_r3 = ip.m + jp.m;
	real_tn ee = ip.e2 + jp.e2;
	real_tn rx = ip.rx - jp.rx;
	real_tn ry = ip.ry - jp.ry;
	real_tn rz = ip.rz - jp.rz;
	real_tn vx = ip.vx - jp.vx;
	real_tn vy = ip.vy - jp.vy;
	real_tn vz = ip.vz - jp.vz;

	real_tn rr = ee;
	rr        += rx * rx + ry * ry + rz * rz;
	real_tn rv = rx * vx + ry * vy + rz * vz;
	real_tn vv = vx * vx + vy * vy + vz * vz;

	real_tn inv_r2 = rsqrt(rr);
	m_r3 *= inv_r2;
	inv_r2 *= inv_r2;
	m_r3 *= 2 * inv_r2;

	real_tn m_r5 = m_r3 * inv_r2;
	m_r3 += vv * inv_r2;
	rv *= eta * rsqrt(m_r3);
	m_r5 += m_r3 * inv_r2;
	m_r3 -= m_r5 * rv;

	m_r3 = (rr > ee) ? (m_r3):(0);

	ip.w2_a = fmax(m_r3, ip.w2_a);
	ip.w2_b += m_r3;
	return ip;
}


#endif	// __TSTEP_KERNEL_COMMON_H__
