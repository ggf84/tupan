#ifndef __SNP_CRK_KERNEL_COMMON_H__
#define __SNP_CRK_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
template<typename I, typename J>
static inline void
p2p_snp_crk_kernel_core(I &ip, J &jp)
// flop count: 153
{
	decltype(ip.rdot) rdot;
	for (auto kdot = 0; kdot < 4; ++kdot) {
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
	auto ra = rdot[0][0] * rdot[2][0]
			+ rdot[0][1] * rdot[2][1]
			+ rdot[0][2] * rdot[2][2];
	auto rj = rdot[0][0] * rdot[3][0]
			+ rdot[0][1] * rdot[3][1]
			+ rdot[0][2] * rdot[3][2];
	auto vv = rdot[1][0] * rdot[1][0]
			+ rdot[1][1] * rdot[1][1]
			+ rdot[1][2] * rdot[1][2];
	auto va = rdot[1][0] * rdot[2][0]
			+ rdot[1][1] * rdot[2][1]
			+ rdot[1][2] * rdot[2][2];

	auto s1 = rv;
	auto s2 = ra + vv;
	auto s3 = rj + 3 * va;

	decltype(rr) inv_r2;
	auto inv_r3 = smoothed_inv_r3_inv_r2(rr, e2, &inv_r2);	// flop count: 5

	inv_r2 *= -3;

	constexpr auto cq21 = static_cast<decltype(rr)>(5.0/3.0);
	constexpr auto cq31 = static_cast<decltype(rr)>(8.0/3.0);
	constexpr auto cq32 = static_cast<decltype(rr)>(7.0/3.0);

	const auto q1 = inv_r2 * (s1);
	const auto q2 = inv_r2 * (s2 + (cq21 * s1) * q1);
	const auto q3 = inv_r2 * (s3 + (cq31 * s2) * q1 + (cq32 * s1) * q2);

	const auto c1 = 3 * q1;
	const auto c2 = 3 * q2;
	const auto c3 = 2 * q1;
	for (auto kdim = 0; kdim < NDIM; ++kdim) {
		rdot[3][kdim] += c1 * rdot[2][kdim];
		rdot[3][kdim] += c2 * rdot[1][kdim];
		rdot[3][kdim] += q3 * rdot[0][kdim];

		rdot[2][kdim] += c3 * rdot[1][kdim];
		rdot[2][kdim] += q2 * rdot[0][kdim];

		rdot[1][kdim] += q1 * rdot[0][kdim];
	}

	{	// i-particle
		const auto m_r3 = jp.m * inv_r3;
		for (auto kdot = 0; kdot < 4; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				ip.adot[kdot][kdim] -= m_r3 * rdot[kdot][kdim];
			}
		}
	}
	{	// j-particle
		const auto m_r3 = ip.m * inv_r3;
		for (auto kdot = 0; kdot < 4; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				jp.adot[kdot][kdim] += m_r3 * rdot[kdot][kdim];
			}
		}
	}
}
#endif

// ----------------------------------------------------------------------------

#define SNP_CRK_IMPLEMENT_STRUCT(N)				\
	typedef struct concat(snp_crk_data, N) {	\
		concat(real_t, N) m, e2;				\
		concat(real_t, N) rdot[4][NDIM];		\
		concat(real_t, N) adot[4][NDIM];		\
	} concat(Snp_Crk_Data, N);

SNP_CRK_IMPLEMENT_STRUCT(1)
#if SIMD > 1
SNP_CRK_IMPLEMENT_STRUCT(SIMD)
#endif
typedef Snp_Crk_Data1 Snp_Crk_Data;


static inline vec(Snp_Crk_Data)
snp_crk_kernel_core(vec(Snp_Crk_Data) ip, Snp_Crk_Data jp)
// flop count: 128
{
	vec(real_t) rdot[4][NDIM];
	#pragma unroll
	for (uint_t kdot = 0; kdot < 4; ++kdot) {
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
	vec(real_t) ra = rdot[0][0] * rdot[2][0]
			+ rdot[0][1] * rdot[2][1]
			+ rdot[0][2] * rdot[2][2];
	vec(real_t) rj = rdot[0][0] * rdot[3][0]
			+ rdot[0][1] * rdot[3][1]
			+ rdot[0][2] * rdot[3][2];
	vec(real_t) vv = rdot[1][0] * rdot[1][0]
			+ rdot[1][1] * rdot[1][1]
			+ rdot[1][2] * rdot[1][2];
	vec(real_t) va = rdot[1][0] * rdot[2][0]
			+ rdot[1][1] * rdot[2][1]
			+ rdot[1][2] * rdot[2][2];

	vec(real_t) s1 = rv;
	vec(real_t) s2 = ra + vv;
	vec(real_t) s3 = rj + 3 * va;

	vec(real_t) inv_r2;
	vec(real_t) m_r3 = jp.m * smoothed_inv_r3_inv_r2(rr, e2, &inv_r2);	// flop count: 6
	inv_r2 = select((vec(real_t))(0), inv_r2, (vec(int_t))(rr > 0));
	m_r3 = select((vec(real_t))(0), m_r3, (vec(int_t))(rr > 0));

	inv_r2 *= -3;

	#define _cq21 ((real_t)(5.0/3.0))
	#define _cq31 ((real_t)(8.0/3.0))
	#define _cq32 ((real_t)(7.0/3.0))

	const vec(real_t) q1 = inv_r2 * (s1);
	const vec(real_t) q2 = inv_r2 * (s2 + (_cq21 * s1) * q1);
	const vec(real_t) q3 = inv_r2 * (s3 + (_cq31 * s2) * q1 + (_cq32 * s1) * q2);

	const vec(real_t) c1 = 3 * q1;
	const vec(real_t) c2 = 3 * q2;
	const vec(real_t) c3 = 2 * q1;

	#pragma unroll
	for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
		rdot[3][kdim] += c1 * rdot[2][kdim];
		rdot[3][kdim] += c2 * rdot[1][kdim];
		rdot[3][kdim] += q3 * rdot[0][kdim];

		rdot[2][kdim] += c3 * rdot[1][kdim];
		rdot[2][kdim] += q2 * rdot[0][kdim];

		rdot[1][kdim] += q1 * rdot[0][kdim];
	}

	#pragma unroll
	for (uint_t kdot = 0; kdot < 4; ++kdot) {
		#pragma unroll
		for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
			ip.adot[kdot][kdim] -= m_r3 * rdot[kdot][kdim];
		}
	}
	return ip;
}


#endif	// __SNP_CRK_KERNEL_COMMON_H__
