#ifndef __PHI_KERNEL_COMMON_H__
#define __PHI_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
template<size_t TILE>
struct Phi_Data_SoA {
	real_t m[TILE];
	real_t e2[TILE];
	real_t rx[TILE];
	real_t ry[TILE];
	real_t rz[TILE];
	real_t phi[TILE];
};

template<size_t TILE>
struct P2P_phi_kernel_core {
	template<typename IP, typename JP>
	void operator()(IP&& ip, JP&& jp) {
		// flop count: 16
		for (size_t i = 0; i < TILE; ++i) {
			#pragma unroll
			for (size_t j = 0; j < TILE; ++j) {
				auto e2 = ip.e2[i] + jp.e2[j];
				auto rx = ip.rx[i] - jp.rx[j];
				auto ry = ip.ry[i] - jp.ry[j];
				auto rz = ip.rz[i] - jp.rz[j];

				auto rr = rx * rx + ry * ry + rz * rz;

				auto inv_r1 = smoothed_inv_r1(rr, e2);	// flop count: 3

				ip.phi[i] -= jp.m[j] * inv_r1;
				jp.phi[j] -= ip.m[i] * inv_r1;
			}
		}
	}

	template<typename P>
	void operator()(P&& p) {
		// flop count: 14
		for (size_t i = 0; i < TILE; ++i) {
			#pragma unroll
			for (size_t j = 0; j < TILE; ++j) {
				auto e2 = p.e2[i] + p.e2[j];
				auto rx = p.rx[i] - p.rx[j];
				auto ry = p.ry[i] - p.ry[j];
				auto rz = p.rz[i] - p.rz[j];

				auto rr = rx * rx + ry * ry + rz * rz;

				auto inv_r1 = smoothed_inv_r1(rr, e2);	// flop count: 3

				inv_r1 = (rr > 0) ? (inv_r1):(0);

				p.phi[i] -= p.m[j] * inv_r1;
			}
		}
	}
};
#endif

// ----------------------------------------------------------------------------

#define PHI_IMPLEMENT_STRUCT(N)				\
	typedef struct concat(phi_data, N) {	\
		concat(real_t, N) m, e2;			\
		concat(real_t, N) rdot[1][NDIM];	\
		concat(real_t, N) phi;				\
	} concat(Phi_Data, N);

PHI_IMPLEMENT_STRUCT(1)
#if SIMD > 1
PHI_IMPLEMENT_STRUCT(SIMD)
#endif
typedef Phi_Data1 Phi_Data;


static inline vec(Phi_Data)
phi_kernel_core(vec(Phi_Data) ip, vec(Phi_Data) jp)
// flop count: 14
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

	vec(real_t) inv_r1 = smoothed_inv_r1(rr, e2);	// flop count: 3
	inv_r1 = select((vec(real_t))(0), inv_r1, (vec(int_t))(rr > 0));

	ip.phi -= jp.m * inv_r1;
	return ip;
}


#endif	// __PHI_KERNEL_COMMON_H__
