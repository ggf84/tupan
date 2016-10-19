#ifndef __TSTEP_KERNEL_COMMON_H__
#define __TSTEP_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


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
		__dt_a[k] = eta / sqrt(p.w2_a[kk]);
		__dt_b[k] = eta / sqrt(p.w2_b[kk]);
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
				auto rr = ip.e2[i] + jp.e2[j];
				auto rx = ip.rx[i] - jp.rx[j];
				auto ry = ip.ry[i] - jp.ry[j];
				auto rz = ip.rz[i] - jp.rz[j];
				auto vx = ip.vx[i] - jp.vx[j];
				auto vy = ip.vy[i] - jp.vy[j];
				auto vz = ip.vz[i] - jp.vz[j];

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
		decltype(p.m) rv, vv, m_r3, inv_r2;
		for (size_t i = 0; i < TILE; ++i) {
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				m_r3[j] = p.m[i] + p.m[j];
				auto rr = p.e2[i] + p.e2[j];
				auto rx = p.rx[i] - p.rx[j];
				auto ry = p.ry[i] - p.ry[j];
				auto rz = p.rz[i] - p.rz[j];
				auto vx = p.vx[i] - p.vx[j];
				auto vy = p.vy[i] - p.vy[j];
				auto vz = p.vz[i] - p.vz[j];

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
				m_r3[j] = (i == j) ? (0):(m_r3[j]);
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				p.w2_a[j] = fmax(m_r3[j], p.w2_a[j]);
				p.w2_b[j] += m_r3[j];
			}
		}
	}
};
#endif

// ----------------------------------------------------------------------------

#define TSTEP_IMPLEMENT_STRUCT(N)			\
	typedef struct concat(tstep_data, N) {	\
		concat(real_t, N) m, e2;			\
		concat(real_t, N) rdot[2][NDIM];	\
		concat(real_t, N) w2[2];			\
	} concat(Tstep_Data, N);

TSTEP_IMPLEMENT_STRUCT(1)
#if SIMD > 1
TSTEP_IMPLEMENT_STRUCT(SIMD)
#endif
typedef Tstep_Data1 Tstep_Data;


static inline vec(Tstep_Data)
tstep_kernel_core(vec(Tstep_Data) ip, vec(Tstep_Data) jp, const real_t eta)
// flop count: 42
{
	vec(real_t) rdot[2][NDIM];
	#pragma unroll
	for (uint_t kdot = 0; kdot < 2; ++kdot) {
		#pragma unroll
		for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
			rdot[kdot][kdim] = ip.rdot[kdot][kdim] - jp.rdot[kdot][kdim];
		}
	}
	vec(real_t) m = ip.m + jp.m;
	vec(real_t) e2 = ip.e2 + jp.e2;

	vec(real_t) rr = rdot[0][0] * rdot[0][0]
			+ rdot[0][1] * rdot[0][1]
			+ rdot[0][2] * rdot[0][2];
	vec(real_t) rv = rdot[0][0] * rdot[1][0]
			+ rdot[0][1] * rdot[1][1]
			+ rdot[0][2] * rdot[1][2];
	vec(real_t) vv = rdot[1][0] * rdot[1][0]
			+ rdot[1][1] * rdot[1][1]
			+ rdot[1][2] * rdot[1][2];

	vec(real_t) inv_r2;
	vec(real_t) m_r3 = 2 * m * smoothed_inv_r3_inv_r2(rr, e2, &inv_r2);	// flop count: 7

	vec(real_t) m_r5 = m_r3 * inv_r2;
	vec(real_t) w2 = m_r3 + vv * inv_r2;
	vec(real_t) gamma = m_r5 + w2 * inv_r2;
	gamma *= (eta * rsqrt(w2));
	w2 -= gamma * rv;

	w2 = select((vec(real_t))(0), w2, (vec(int_t))(rr > 0));

	ip.w2[0] = fmax(w2, ip.w2[0]);
	ip.w2[1] += w2;
	return ip;
}


#endif	// __TSTEP_KERNEL_COMMON_H__
