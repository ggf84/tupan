#ifndef __PNACC_KERNEL_COMMON_H__
#define __PNACC_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"
#include "pn_terms.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
template<typename I, typename J, typename PARAM>
static inline void
p2p_pnacc_kernel_core(I &ip, J &jp, const PARAM clight)
// flop count: 48 + ???
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
	auto vv = rdot[1][0] * rdot[1][0]
			+ rdot[1][1] * rdot[1][1]
			+ rdot[1][2] * rdot[1][2];

	decltype(rr) inv_r1;
	auto inv_r2 = smoothed_inv_r2_inv_r1(rr, e2, &inv_r1);	// flop count: 4

	{	// i-particle
		auto pn = p2p_pnterms(
			ip.m, ip.rdot[1][0], ip.rdot[1][1], ip.rdot[1][2],
			jp.m, jp.rdot[1][0], jp.rdot[1][1], jp.rdot[1][2],
			rdot[0][0], rdot[0][1], rdot[0][2],
			rdot[1][0], rdot[1][1], rdot[1][2],
			vv, inv_r1, inv_r2, clight
		);	// flop count: ???

		for (auto kdim = 0; kdim < NDIM; ++kdim) {
			ip.pnacc[kdim] += (pn.a * rdot[0][kdim] + pn.b * rdot[1][kdim]);
		}
	}
	{	// j-particle
		auto pn = p2p_pnterms(
			jp.m, jp.rdot[1][0], jp.rdot[1][1], jp.rdot[1][2],
			ip.m, ip.rdot[1][0], ip.rdot[1][1], ip.rdot[1][2],
			-rdot[0][0], -rdot[0][1], -rdot[0][2],
			-rdot[1][0], -rdot[1][1], -rdot[1][2],
			vv, inv_r1, inv_r2, clight
		);	// flop count: ???

		for (auto kdim = 0; kdim < NDIM; ++kdim) {
			jp.pnacc[kdim] -= (pn.a * rdot[0][kdim] + pn.b * rdot[1][kdim]);
		}
	}
}
#endif

// ----------------------------------------------------------------------------

#define PNACC_IMPLEMENT_STRUCT(N)			\
	typedef struct concat(pnacc_data, N) {	\
		concat(real_t, N) m, e2;			\
		concat(real_t, N) rdot[2][NDIM];	\
		concat(real_t, N) pnacc[NDIM];		\
	} concat(PNAcc_Data, N);

PNACC_IMPLEMENT_STRUCT(1)
#if SIMD > 1
PNACC_IMPLEMENT_STRUCT(SIMD)
#endif
typedef PNAcc_Data1 PNAcc_Data;


static inline vec(PNAcc_Data)
pnacc_kernel_core(vec(PNAcc_Data) ip, vec(PNAcc_Data) jp, const CLIGHT clight)
// flop count: 36+???
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
	vec(real_t) vv = rdot[1][0] * rdot[1][0]
			+ rdot[1][1] * rdot[1][1]
			+ rdot[1][2] * rdot[1][2];

	vec(real_t) inv_r1;
	vec(real_t) inv_r2 = smoothed_inv_r2_inv_r1(rr, e2, &inv_r1);	// flop count: 4
	inv_r1 = select((vec(real_t))(0), inv_r1, (vec(int_t))(rr > 0));
	inv_r2 = select((vec(real_t))(0), inv_r2, (vec(int_t))(rr > 0));

	PN pn = p2p_pnterms(
		ip.m, ip.rdot[1][0], ip.rdot[1][1], ip.rdot[1][2],
		jp.m, jp.rdot[1][0], jp.rdot[1][1], jp.rdot[1][2],
		rdot[0][0], rdot[0][1], rdot[0][2],
		rdot[1][0], rdot[1][1], rdot[1][2],
		vv, inv_r1, inv_r2, clight
	);	// flop count: ???

	#pragma unroll
	for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
		ip.pnacc[kdim] += (pn.a * rdot[0][kdim] + pn.b * rdot[1][kdim]);
	}
	return ip;
}


#endif	// __PNACC_KERNEL_COMMON_H__
