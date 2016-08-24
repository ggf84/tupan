#ifndef __TSTEP_KERNEL_COMMON_H__
#define __TSTEP_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
template<typename I, typename J, typename PARAM>
static inline void
p2p_tstep_kernel_core(I &ip, J &jp, const PARAM eta)
// flop count: 43
{
	decltype(ip.rdot) rdot;
	for (auto kdot = 0; kdot < 2; ++kdot) {
		for (auto kdim = 0; kdim < NDIM; ++kdim) {
			rdot[kdot][kdim] = ip.rdot[kdot][kdim] - jp.rdot[kdot][kdim];
		}
	}
	auto m = ip.m + jp.m;
	auto e2 = ip.e2 + jp.e2;

	auto rr = rdot[0][0] * rdot[0][0]
			+ rdot[0][1] * rdot[0][1]
			+ rdot[0][2] * rdot[0][2];
	auto rv = rdot[0][0] * rdot[1][0]
			+ rdot[0][1] * rdot[1][1]
			+ rdot[0][2] * rdot[1][2];
	auto vv = rdot[1][0] * rdot[1][0]
			+ rdot[1][1] * rdot[1][1]
			+ rdot[1][2] * rdot[1][2];

	decltype(rr) inv_r2;
	auto m_r3 = 2 * m * smoothed_inv_r3_inv_r2(rr, e2, &inv_r2);	// flop count: 7

	auto m_r5 = m_r3 * inv_r2;
	auto w2 = m_r3 + vv * inv_r2;
	auto gamma = m_r5 + w2 * inv_r2;
	gamma *= (eta * rsqrt(w2));
	w2 -= gamma * rv;

	ip.w2[0] = fmax(w2, ip.w2[0]);
	ip.w2[1] += w2;
	jp.w2[0] = fmax(w2, jp.w2[0]);
	jp.w2[1] += w2;
}
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
tstep_kernel_core(vec(Tstep_Data) ip, Tstep_Data jp, const real_t eta)
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
