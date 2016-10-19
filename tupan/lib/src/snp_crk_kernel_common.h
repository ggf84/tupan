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

template<size_t TILE, typename T = Snp_Crk_Data_SoA<TILE>>
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
		p.ax[kk] = __rdot[(2*NDIM+0)*n + k];
		p.ay[kk] = __rdot[(2*NDIM+1)*n + k];
		p.az[kk] = __rdot[(2*NDIM+2)*n + k];
		p.jx[kk] = __rdot[(3*NDIM+0)*n + k];
		p.jy[kk] = __rdot[(3*NDIM+1)*n + k];
		p.jz[kk] = __rdot[(3*NDIM+2)*n + k];
	}
	return part;
}

template<size_t TILE, typename PART>
void commit(const uint_t n, const PART& part, real_t __adot[])
{
	for (size_t k = 0; k < n; ++k) {
		auto kk = k%TILE;
		auto& p = part[k/TILE];
		__adot[(0*NDIM+0)*n + k] = p.Ax[kk];
		__adot[(0*NDIM+1)*n + k] = p.Ay[kk];
		__adot[(0*NDIM+2)*n + k] = p.Az[kk];
		__adot[(1*NDIM+0)*n + k] = p.Jx[kk];
		__adot[(1*NDIM+1)*n + k] = p.Jy[kk];
		__adot[(1*NDIM+2)*n + k] = p.Jz[kk];
		__adot[(2*NDIM+0)*n + k] = p.Sx[kk];
		__adot[(2*NDIM+1)*n + k] = p.Sy[kk];
		__adot[(2*NDIM+2)*n + k] = p.Sz[kk];
		__adot[(3*NDIM+0)*n + k] = p.Cx[kk];
		__adot[(3*NDIM+1)*n + k] = p.Cy[kk];
		__adot[(3*NDIM+2)*n + k] = p.Cz[kk];
	}
}

template<size_t TILE>
struct P2P_snp_crk_kernel_core {
	template<typename IP, typename JP>
	void operator()(IP&& ip, JP&& jp) {
		// flop count: 153
		decltype(jp.m) rx, ry, rz, vx, vy, vz;
		decltype(jp.m) ax, ay, az, jx, jy, jz;
		decltype(jp.m) inv_r3, im_r3, jm_r3;
		for (size_t i = 0; i < TILE; ++i) {
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto rr = ip.e2[i] + jp.e2[j];
				rx[j] = ip.rx[i] - jp.rx[j];
				ry[j] = ip.ry[i] - jp.ry[j];
				rz[j] = ip.rz[i] - jp.rz[j];
				vx[j] = ip.vx[i] - jp.vx[j];
				vy[j] = ip.vy[i] - jp.vy[j];
				vz[j] = ip.vz[i] - jp.vz[j];
				ax[j] = ip.ax[i] - jp.ax[j];
				ay[j] = ip.ay[i] - jp.ay[j];
				az[j] = ip.az[i] - jp.az[j];
				jx[j] = ip.jx[i] - jp.jx[j];
				jy[j] = ip.jy[i] - jp.jy[j];
				jz[j] = ip.jz[i] - jp.jz[j];

				rr += rx[j] * rx[j] + ry[j] * ry[j] + rz[j] * rz[j];

				inv_r3[j] = rsqrt(rr);
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto inv_r2 = inv_r3[j] * inv_r3[j];
				inv_r3[j] *= inv_r2;
				inv_r2 *= -3;

				auto s1 = rx[j] * vx[j] + ry[j] * vy[j] + rz[j] * vz[j];
				auto s2 = vx[j] * vx[j] + vy[j] * vy[j] + vz[j] * vz[j];
				auto s3 = vx[j] * ax[j] + vy[j] * ay[j] + vz[j] * az[j];
				s3 *= 3;
				s2 += rx[j] * ax[j] + ry[j] * ay[j] + rz[j] * az[j];
				s3 += rx[j] * jx[j] + ry[j] * jy[j] + rz[j] * jz[j];

				constexpr auto cq21 = static_cast<decltype(s1)>(5.0/3.0);
				constexpr auto cq31 = static_cast<decltype(s1)>(8.0/3.0);
				constexpr auto cq32 = static_cast<decltype(s1)>(7.0/3.0);

				const auto q1 = inv_r2 * (s1);
				const auto q2 = inv_r2 * (s2 + (cq21 * s1) * q1);
				const auto q3 = inv_r2 * (s3 + (cq31 * s2) * q1 + (cq32 * s1) * q2);

				const auto b3 = 3 * q1;
				const auto c3 = 3 * q2;
				const auto c2 = 2 * q1;
				jx[j] += b3 * ax[j] + c3 * vx[j] + q3 * rx[j];
				jy[j] += b3 * ay[j] + c3 * vy[j] + q3 * ry[j];
				jz[j] += b3 * az[j] + c3 * vz[j] + q3 * rz[j];

				ax[j] += c2 * vx[j] + q2 * rx[j];
				ay[j] += c2 * vy[j] + q2 * ry[j];
				az[j] += c2 * vz[j] + q2 * rz[j];

				vx[j] += q1 * rx[j];
				vy[j] += q1 * ry[j];
				vz[j] += q1 * rz[j];

				im_r3[j] = ip.m[i] * inv_r3[j];
				jm_r3[j] = jp.m[j] * inv_r3[j];
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				jp.Ax[j] += im_r3[j] * rx[j];
				jp.Ay[j] += im_r3[j] * ry[j];
				jp.Az[j] += im_r3[j] * rz[j];
				jp.Jx[j] += im_r3[j] * vx[j];
				jp.Jy[j] += im_r3[j] * vy[j];
				jp.Jz[j] += im_r3[j] * vz[j];
				jp.Sx[j] += im_r3[j] * ax[j];
				jp.Sy[j] += im_r3[j] * ay[j];
				jp.Sz[j] += im_r3[j] * az[j];
				jp.Cx[j] += im_r3[j] * jx[j];
				jp.Cy[j] += im_r3[j] * jy[j];
				jp.Cz[j] += im_r3[j] * jz[j];
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				ip.Ax[i] -= jm_r3[j] * rx[j];
				ip.Ay[i] -= jm_r3[j] * ry[j];
				ip.Az[i] -= jm_r3[j] * rz[j];
				ip.Jx[i] -= jm_r3[j] * vx[j];
				ip.Jy[i] -= jm_r3[j] * vy[j];
				ip.Jz[i] -= jm_r3[j] * vz[j];
				ip.Sx[i] -= jm_r3[j] * ax[j];
				ip.Sy[i] -= jm_r3[j] * ay[j];
				ip.Sz[i] -= jm_r3[j] * az[j];
				ip.Cx[i] -= jm_r3[j] * jx[j];
				ip.Cy[i] -= jm_r3[j] * jy[j];
				ip.Cz[i] -= jm_r3[j] * jz[j];
			}
		}
	}

	template<typename P>
	void operator()(P&& p) {
		// flop count: 128
		decltype(p.m) rx, ry, rz, vx, vy, vz;
		decltype(p.m) ax, ay, az, jx, jy, jz;
		decltype(p.m) inv_r3, im_r3;
		for (size_t i = 0; i < TILE; ++i) {
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto rr = p.e2[i] + p.e2[j];
				rx[j] = p.rx[i] - p.rx[j];
				ry[j] = p.ry[i] - p.ry[j];
				rz[j] = p.rz[i] - p.rz[j];
				vx[j] = p.vx[i] - p.vx[j];
				vy[j] = p.vy[i] - p.vy[j];
				vz[j] = p.vz[i] - p.vz[j];
				ax[j] = p.ax[i] - p.ax[j];
				ay[j] = p.ay[i] - p.ay[j];
				az[j] = p.az[i] - p.az[j];
				jx[j] = p.jx[i] - p.jx[j];
				jy[j] = p.jy[i] - p.jy[j];
				jz[j] = p.jz[i] - p.jz[j];

				rr += rx[j] * rx[j] + ry[j] * ry[j] + rz[j] * rz[j];

				inv_r3[j] = rsqrt(rr);
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				inv_r3[j] = (i == j) ? (0):(inv_r3[j]);
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto inv_r2 = inv_r3[j] * inv_r3[j];
				inv_r3[j] *= inv_r2;
				inv_r2 *= -3;

				auto s1 = rx[j] * vx[j] + ry[j] * vy[j] + rz[j] * vz[j];
				auto s2 = vx[j] * vx[j] + vy[j] * vy[j] + vz[j] * vz[j];
				auto s3 = vx[j] * ax[j] + vy[j] * ay[j] + vz[j] * az[j];
				s3 *= 3;
				s2 += rx[j] * ax[j] + ry[j] * ay[j] + rz[j] * az[j];
				s3 += rx[j] * jx[j] + ry[j] * jy[j] + rz[j] * jz[j];

				constexpr auto cq21 = static_cast<decltype(s1)>(5.0/3.0);
				constexpr auto cq31 = static_cast<decltype(s1)>(8.0/3.0);
				constexpr auto cq32 = static_cast<decltype(s1)>(7.0/3.0);

				const auto q1 = inv_r2 * (s1);
				const auto q2 = inv_r2 * (s2 + (cq21 * s1) * q1);
				const auto q3 = inv_r2 * (s3 + (cq31 * s2) * q1 + (cq32 * s1) * q2);

				const auto b3 = 3 * q1;
				const auto c3 = 3 * q2;
				const auto c2 = 2 * q1;
				jx[j] += b3 * ax[j] + c3 * vx[j] + q3 * rx[j];
				jy[j] += b3 * ay[j] + c3 * vy[j] + q3 * ry[j];
				jz[j] += b3 * az[j] + c3 * vz[j] + q3 * rz[j];

				ax[j] += c2 * vx[j] + q2 * rx[j];
				ay[j] += c2 * vy[j] + q2 * ry[j];
				az[j] += c2 * vz[j] + q2 * rz[j];

				vx[j] += q1 * rx[j];
				vy[j] += q1 * ry[j];
				vz[j] += q1 * rz[j];

				im_r3[j] = p.m[i] * inv_r3[j];
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				p.Ax[j] += im_r3[j] * rx[j];
				p.Ay[j] += im_r3[j] * ry[j];
				p.Az[j] += im_r3[j] * rz[j];
				p.Jx[j] += im_r3[j] * vx[j];
				p.Jy[j] += im_r3[j] * vy[j];
				p.Jz[j] += im_r3[j] * vz[j];
				p.Sx[j] += im_r3[j] * ax[j];
				p.Sy[j] += im_r3[j] * ay[j];
				p.Sz[j] += im_r3[j] * az[j];
				p.Cx[j] += im_r3[j] * jx[j];
				p.Cy[j] += im_r3[j] * jy[j];
				p.Cz[j] += im_r3[j] * jz[j];
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
