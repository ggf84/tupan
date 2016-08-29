#ifndef __ACC_JRK_KERNEL_COMMON_H__
#define __ACC_JRK_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
template<typename I, typename J>
static inline void
p2p_acc_jrk_kernel_core(I &ip, J &jp)
// flop count: 56
{
	decltype(ip.rdot) rdot;
	for (auto kdot = 0; kdot < 2; ++kdot) {
		for (auto kdim = 0; kdim < NDIM; ++kdim) {
			rdot[kdot][kdim] = ip.rdot[kdot][kdim] - jp.rdot[kdot][kdim];
		}
	}
	auto e2 = ip.e2 + jp.e2;

	auto rr = rdot[0][0] * rdot[0][0]
			+ rdot[0][1] * rdot[0][1]
			+ rdot[0][2] * rdot[0][2];
	auto rv = rdot[0][0] * rdot[1][0]
			+ rdot[0][1] * rdot[1][1]
			+ rdot[0][2] * rdot[1][2];

	auto s1 = rv;

	decltype(rr) inv_r2;
	auto inv_r3 = smoothed_inv_r3_inv_r2(rr, e2, &inv_r2);	// flop count: 5

	inv_r2 *= -3;

	const auto q1 = inv_r2 * (s1);

	for (auto kdim = 0; kdim < NDIM; ++kdim) {
		rdot[1][kdim] += q1 * rdot[0][kdim];
	}

	{	// i-particle
		const auto m_r3 = jp.m * inv_r3;
		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				ip.adot[kdot][kdim] -= m_r3 * rdot[kdot][kdim];
			}
		}
	}
	{	// j-particle
		const auto m_r3 = ip.m * inv_r3;
		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				jp.adot[kdot][kdim] += m_r3 * rdot[kdot][kdim];
			}
		}
	}
}
#endif

// ----------------------------------------------------------------------------

#define ACC_JRK_IMPLEMENT_STRUCT(N)				\
	typedef struct concat(acc_jrk_data, N) {	\
		concat(real_t, N) m, e2;				\
		concat(real_t, N) rdot[2][NDIM];		\
		concat(real_t, N) adot[2][NDIM];		\
	} concat(Acc_Jrk_Data, N);

ACC_JRK_IMPLEMENT_STRUCT(1)
#if SIMD > 1
ACC_JRK_IMPLEMENT_STRUCT(SIMD)
#endif
typedef Acc_Jrk_Data1 Acc_Jrk_Data;


static inline vec(Acc_Jrk_Data)
acc_jrk_kernel_core(vec(Acc_Jrk_Data) ip, vec(Acc_Jrk_Data) jp)
// flop count: 43
{
	vec(real_t) rdot[2][NDIM];
	#pragma unroll
	for (uint_t kdot = 0; kdot < 2; ++kdot) {
		#pragma unroll
		for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
			rdot[kdot][kdim] = ip.rdot[kdot][kdim] - jp.rdot[kdot][kdim];
		}
	}
	vec(real_t) e2 = ip.e2 + jp.e2;

	vec(real_t) rr = rdot[0][0] * rdot[0][0]
			+ rdot[0][1] * rdot[0][1]
			+ rdot[0][2] * rdot[0][2];
	vec(real_t) rv = rdot[0][0] * rdot[1][0]
			+ rdot[0][1] * rdot[1][1]
			+ rdot[0][2] * rdot[1][2];

	vec(real_t) s1 = rv;

	vec(real_t) inv_r2;
	vec(real_t) m_r3 = jp.m * smoothed_inv_r3_inv_r2(rr, e2, &inv_r2);	// flop count: 6
	inv_r2 = select((vec(real_t))(0), inv_r2, (vec(int_t))(rr > 0));
	m_r3 = select((vec(real_t))(0), m_r3, (vec(int_t))(rr > 0));

	inv_r2 *= -3;

	const vec(real_t) q1 = inv_r2 * (s1);

	#pragma unroll
	for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
		rdot[1][kdim] += q1 * rdot[0][kdim];
	}

	#pragma unroll
	for (uint_t kdot = 0; kdot < 2; ++kdot) {
		#pragma unroll
		for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
			ip.adot[kdot][kdim] -= m_r3 * rdot[kdot][kdim];
		}
	}
	return ip;
}


#endif	// __ACC_JRK_KERNEL_COMMON_H__
