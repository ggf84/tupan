#ifndef __ACC_KERNEL_COMMON_H__
#define __ACC_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
template<size_t TILE>
struct Acc_Data_SoA {
	real_t m[TILE];
	real_t e2[TILE];
	real_t rx[TILE];
	real_t ry[TILE];
	real_t rz[TILE];
	real_t ax[TILE];
	real_t ay[TILE];
	real_t az[TILE];
};

template<size_t TILE, typename T = Acc_Data_SoA<TILE>>
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
	}
	return part;
}

template<size_t TILE, typename PART>
void commit(const uint_t n, const PART& part, real_t __adot[])
{
	for (size_t k = 0; k < n; ++k) {
		auto kk = k%TILE;
		auto& p = part[k/TILE];
		__adot[(0*NDIM+0)*n + k] = p.ax[kk];
		__adot[(0*NDIM+1)*n + k] = p.ay[kk];
		__adot[(0*NDIM+2)*n + k] = p.az[kk];
	}
}

template<size_t TILE>
struct P2P_acc_kernel_core {
	template<typename IP, typename JP>
	void operator()(IP&& ip, JP&& jp) {
		// flop count: 28
		for (size_t i = 0; i < TILE; ++i) {
			#pragma unroll
			for (size_t j = 0; j < TILE; ++j) {
				auto rr = ip.e2[i] + jp.e2[j];
				auto rx = ip.rx[i] - jp.rx[j];
				auto ry = ip.ry[i] - jp.ry[j];
				auto rz = ip.rz[i] - jp.rz[j];

				rr += rx * rx + ry * ry + rz * rz;

				auto inv_r3 = rsqrt(rr);
				inv_r3 *= inv_r3 * inv_r3;

				// i-particle
				auto jm_r3 = jp.m[j] * inv_r3;
				ip.ax[i] -= jm_r3 * rx;
				ip.ay[i] -= jm_r3 * ry;
				ip.az[i] -= jm_r3 * rz;

				// j-particle
				auto im_r3 = ip.m[i] * inv_r3;
				jp.ax[j] += im_r3 * rx;
				jp.ay[j] += im_r3 * ry;
				jp.az[j] += im_r3 * rz;
			}
		}
	}

	template<typename P>
	void operator()(P&& p) {
		// flop count: 21
		for (size_t i = 0; i < TILE; ++i) {
			#pragma unroll
			for (size_t j = 0; j < TILE; ++j) {
				auto rr = p.e2[i] + p.e2[j];
				auto rx = p.rx[i] - p.rx[j];
				auto ry = p.ry[i] - p.ry[j];
				auto rz = p.rz[i] - p.rz[j];

				rr += rx * rx + ry * ry + rz * rz;

				auto inv_r3 = rsqrt(rr);
				inv_r3 = (i != j) ? (inv_r3):(0);
				inv_r3 *= inv_r3 * inv_r3;

				auto jm_r3 = p.m[j] * inv_r3;
				p.ax[i] -= jm_r3 * rx;
				p.ay[i] -= jm_r3 * ry;
				p.az[i] -= jm_r3 * rz;
			}
		}
	}
};
#endif

// ----------------------------------------------------------------------------

#define ACC_IMPLEMENT_STRUCT(N)				\
	typedef struct concat(acc_data, N) {	\
		concat(real_t, N) m, e2;			\
		concat(real_t, N) rdot[1][NDIM];	\
		concat(real_t, N) adot[1][NDIM];	\
	} concat(Acc_Data, N);

ACC_IMPLEMENT_STRUCT(1)
#if SIMD > 1
ACC_IMPLEMENT_STRUCT(SIMD)
#endif
typedef Acc_Data1 Acc_Data;


static inline vec(Acc_Data)
acc_kernel_core(vec(Acc_Data) ip, vec(Acc_Data) jp)
// flop count: 21
{
	vec(real_t) rdot[1][NDIM];
	#pragma unroll
	for (uint_t kdot = 0; kdot < 1; ++kdot) {
		#pragma unroll
		for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
			rdot[kdot][kdim] = ip.rdot[kdot][kdim] - jp.rdot[kdot][kdim];
		}
	}
	vec(real_t) e2 = ip.e2 + jp.e2;

	vec(real_t) rr = rdot[0][0] * rdot[0][0]
			+ rdot[0][1] * rdot[0][1]
			+ rdot[0][2] * rdot[0][2];

	vec(real_t) m_r3 = jp.m * smoothed_inv_r3(rr, e2);	// flop count: 6
	m_r3 = select((vec(real_t))(0), m_r3, (vec(int_t))(rr > 0));

	#pragma unroll
	for (uint_t kdot = 0; kdot < 1; ++kdot) {
		#pragma unroll
		for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
			ip.adot[kdot][kdim] -= m_r3 * rdot[kdot][kdim];
		}
	}
	return ip;
}


#endif	// __ACC_KERNEL_COMMON_H__
