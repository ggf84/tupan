#ifndef __PHI_KERNEL_COMMON_H__
#define __PHI_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
template<typename I, typename J>
static inline void
p2p_phi_kernel_core(I &ip, J &jp)
// flop count: 16
{
	decltype(ip.rdot) rdot;
	for (auto kdot = 0; kdot < 1; ++kdot) {
		for (auto kdim = 0; kdim < NDIM; ++kdim) {
			rdot[kdot][kdim] = ip.rdot[kdot][kdim] - jp.rdot[kdot][kdim];
		}
	}
	auto e2 = ip.e2 + jp.e2;

	auto rr = rdot[0][0] * rdot[0][0]
			+ rdot[0][1] * rdot[0][1]
			+ rdot[0][2] * rdot[0][2];

	auto inv_r1 = smoothed_inv_r1(rr, e2);	// flop count: 3

	ip.phi -= jp.m * inv_r1;
	jp.phi -= ip.m * inv_r1;
}
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
phi_kernel_core(vec(Phi_Data) ip, Phi_Data jp)
// flop count: 14
{
	vec(real_t) rdot[1][NDIM];
	for (uint_t kdot = 0; kdot < 1; ++kdot) {
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
