#ifndef __SNP_CRK_KERNEL_COMMON_H__
#define __SNP_CRK_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
template<size_t TILE>
struct Snp_Crk_Data_SoA {
	real_t m[TILE];
	real_t e2[TILE];
	real_t rx[TILE];
	real_t ry[TILE];
	real_t rz[TILE];
	real_t vx[TILE];
	real_t vy[TILE];
	real_t vz[TILE];
	real_t ax[TILE];
	real_t ay[TILE];
	real_t az[TILE];
	real_t jx[TILE];
	real_t jy[TILE];
	real_t jz[TILE];
	real_t Ax[TILE];
	real_t Ay[TILE];
	real_t Az[TILE];
	real_t Jx[TILE];
	real_t Jy[TILE];
	real_t Jz[TILE];
	real_t Sx[TILE];
	real_t Sy[TILE];
	real_t Sz[TILE];
	real_t Cx[TILE];
	real_t Cy[TILE];
	real_t Cz[TILE];
};

template<size_t TILE>
struct P2P_snp_crk_kernel_core {
	template<typename IP, typename JP>
	void operator()(IP&& ip, JP&& jp) {
		// flop count: 153
		for (size_t i = 0; i < TILE; ++i) {
			#pragma unroll
			for (size_t j = 0; j < TILE; ++j) {
				auto e2 = ip.e2[i] + jp.e2[j];
				auto rx = ip.rx[i] - jp.rx[j];
				auto ry = ip.ry[i] - jp.ry[j];
				auto rz = ip.rz[i] - jp.rz[j];
				auto vx = ip.vx[i] - jp.vx[j];
				auto vy = ip.vy[i] - jp.vy[j];
				auto vz = ip.vz[i] - jp.vz[j];
				auto ax = ip.ax[i] - jp.ax[j];
				auto ay = ip.ay[i] - jp.ay[j];
				auto az = ip.az[i] - jp.az[j];
				auto jx = ip.jx[i] - jp.jx[j];
				auto jy = ip.jy[i] - jp.jy[j];
				auto jz = ip.jz[i] - jp.jz[j];

				auto rr = rx * rx + ry * ry + rz * rz;
				auto rv = rx * vx + ry * vy + rz * vz;
				auto ra = rx * ax + ry * ay + rz * az;
				auto rj = rx * jx + ry * jy + rz * jz;
				auto vv = vx * vx + vy * vy + vz * vz;
				auto va = vx * ax + vy * ay + vz * az;

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
				jx += c1 * ax;
				jy += c1 * ay;
				jz += c1 * az;
				jx += c2 * vx;
				jy += c2 * vy;
				jz += c2 * vz;
				jx += q3 * rx;
				jy += q3 * ry;
				jz += q3 * rz;

				ax += c3 * vx;
				ay += c3 * vy;
				az += c3 * vz;
				ax += q2 * rx;
				ay += q2 * ry;
				az += q2 * rz;

				vx += q1 * rx;
				vy += q1 * ry;
				vz += q1 * rz;

				{	// i-particle
					auto m_r3 = jp.m[j] * inv_r3;
					ip.Ax[i] -= m_r3 * rx;
					ip.Ay[i] -= m_r3 * ry;
					ip.Az[i] -= m_r3 * rz;
					ip.Jx[i] -= m_r3 * vx;
					ip.Jy[i] -= m_r3 * vy;
					ip.Jz[i] -= m_r3 * vz;
					ip.Sx[i] -= m_r3 * ax;
					ip.Sy[i] -= m_r3 * ay;
					ip.Sz[i] -= m_r3 * az;
					ip.Cx[i] -= m_r3 * jx;
					ip.Cy[i] -= m_r3 * jy;
					ip.Cz[i] -= m_r3 * jz;
				}
				{	// j-particle
					auto m_r3 = ip.m[i] * inv_r3;
					jp.Ax[j] += m_r3 * rx;
					jp.Ay[j] += m_r3 * ry;
					jp.Az[j] += m_r3 * rz;
					jp.Jx[j] += m_r3 * vx;
					jp.Jy[j] += m_r3 * vy;
					jp.Jz[j] += m_r3 * vz;
					jp.Sx[j] += m_r3 * ax;
					jp.Sy[j] += m_r3 * ay;
					jp.Sz[j] += m_r3 * az;
					jp.Cx[j] += m_r3 * jx;
					jp.Cy[j] += m_r3 * jy;
					jp.Cz[j] += m_r3 * jz;
				}
			}
		}
	}

	template<typename P>
	void operator()(P&& p) {
		// flop count: 128
		for (size_t i = 0; i < TILE; ++i) {
			#pragma unroll
			for (size_t j = 0; j < TILE; ++j) {
				auto e2 = p.e2[i] + p.e2[j];
				auto rx = p.rx[i] - p.rx[j];
				auto ry = p.ry[i] - p.ry[j];
				auto rz = p.rz[i] - p.rz[j];
				auto vx = p.vx[i] - p.vx[j];
				auto vy = p.vy[i] - p.vy[j];
				auto vz = p.vz[i] - p.vz[j];
				auto ax = p.ax[i] - p.ax[j];
				auto ay = p.ay[i] - p.ay[j];
				auto az = p.az[i] - p.az[j];
				auto jx = p.jx[i] - p.jx[j];
				auto jy = p.jy[i] - p.jy[j];
				auto jz = p.jz[i] - p.jz[j];

				auto rr = rx * rx + ry * ry + rz * rz;
				auto rv = rx * vx + ry * vy + rz * vz;
				auto ra = rx * ax + ry * ay + rz * az;
				auto rj = rx * jx + ry * jy + rz * jz;
				auto vv = vx * vx + vy * vy + vz * vz;
				auto va = vx * ax + vy * ay + vz * az;

				auto s1 = rv;
				auto s2 = ra + vv;
				auto s3 = rj + 3 * va;

				decltype(rr) inv_r2;
				auto inv_r3 = smoothed_inv_r3_inv_r2(rr, e2, &inv_r2);	// flop count: 5

				inv_r2 = (rr > 0) ? (inv_r2):(0);
				inv_r3 = (rr > 0) ? (inv_r3):(0);

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
				jx += c1 * ax;
				jy += c1 * ay;
				jz += c1 * az;
				jx += c2 * vx;
				jy += c2 * vy;
				jz += c2 * vz;
				jx += q3 * rx;
				jy += q3 * ry;
				jz += q3 * rz;

				ax += c3 * vx;
				ay += c3 * vy;
				az += c3 * vz;
				ax += q2 * rx;
				ay += q2 * ry;
				az += q2 * rz;

				vx += q1 * rx;
				vy += q1 * ry;
				vz += q1 * rz;

				auto m_r3 = p.m[j] * inv_r3;
				p.Ax[i] -= m_r3 * rx;
				p.Ay[i] -= m_r3 * ry;
				p.Az[i] -= m_r3 * rz;
				p.Jx[i] -= m_r3 * vx;
				p.Jy[i] -= m_r3 * vy;
				p.Jz[i] -= m_r3 * vz;
				p.Sx[i] -= m_r3 * ax;
				p.Sy[i] -= m_r3 * ay;
				p.Sz[i] -= m_r3 * az;
				p.Cx[i] -= m_r3 * jx;
				p.Cy[i] -= m_r3 * jy;
				p.Cz[i] -= m_r3 * jz;
			}
		}
	}
};
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
snp_crk_kernel_core(vec(Snp_Crk_Data) ip, vec(Snp_Crk_Data) jp)
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
