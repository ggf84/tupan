#ifndef __ACC_KERNEL_COMMON_H__
#define __ACC_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
template<typename I, typename J>
static inline void
p2p_acc_kernel_core(I &ip, J &jp)
// flop count: 28
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

	auto inv_r3 = smoothed_inv_r3(rr, e2);	// flop count: 5

	{	// i-particle
		auto m_r3 = jp.m * inv_r3;
		for (auto kdot = 0; kdot < 1; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				ip.adot[kdot][kdim] -= m_r3 * rdot[kdot][kdim];
			}
		}
	}
	{	// j-particle
		auto m_r3 = ip.m * inv_r3;
		for (auto kdot = 0; kdot < 1; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				jp.adot[kdot][kdim] += m_r3 * rdot[kdot][kdim];
			}
		}
	}
}
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
acc_kernel_core(vec(Acc_Data) ip, Acc_Data jp)
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
