#ifndef __SNP_CRK_KERNEL_COMMON_H__
#define __SNP_CRK_KERNEL_COMMON_H__


#include "common.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
namespace Snp_Crk {

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
auto setup(
	const uint_t n,
	const real_t __m[],
	const real_t __e2[],
	const real_t __rdot[])
{
	auto ntiles = (n + TILE - 1) / TILE;
	vector<Snp_Crk_Data_SoA<TILE>> part(ntiles);
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
				auto ee = ip.e2[i] + jp.e2[j];
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

				auto rr = ee;
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
		for (size_t i = 0; i < TILE; ++i) {
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto ee = p.e2[i] + p.e2[j];
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

				auto rr = ee;
				rr += rx * rx + ry * ry + rz * rz;

				auto inv_r3 = rsqrt(rr);
				inv_r3 = (rr > ee) ? (inv_r3):(0);
				auto inv_r2 = inv_r3 * inv_r3;
				inv_r3 *= inv_r2;
				inv_r2 *= -3;

				auto s1 = rx * vx + ry * vy + rz * vz;
				auto s2 = vx * vx + vy * vy + vz * vz;
				auto s3 = vx * ax + vy * ay + vz * az;
				s3 *= 3;
				s2 += rx * ax + ry * ay + rz * az;
				s3 += rx * jx + ry * jy + rz * jz;

				constexpr auto cq21 = static_cast<decltype(s1)>(5.0/3.0);
				constexpr auto cq31 = static_cast<decltype(s1)>(8.0/3.0);
				constexpr auto cq32 = static_cast<decltype(s1)>(7.0/3.0);

				const auto q1 = inv_r2 * (s1);
				const auto q2 = inv_r2 * (s2 + (cq21 * s1) * q1);
				const auto q3 = inv_r2 * (s3 + (cq31 * s2) * q1 + (cq32 * s1) * q2);

				const auto b3 = 3 * q1;
				const auto c3 = 3 * q2;
				const auto c2 = 2 * q1;

				jx += b3 * ax + c3 * vx + q3 * rx;
				jy += b3 * ay + c3 * vy + q3 * ry;
				jz += b3 * az + c3 * vz + q3 * rz;

				ax += c2 * vx + q2 * rx;
				ay += c2 * vy + q2 * ry;
				az += c2 * vz + q2 * rz;

				vx += q1 * rx;
				vy += q1 * ry;
				vz += q1 * rz;

				auto im_r3 = p.m[i] * inv_r3;

				p.Ax[j] += im_r3 * rx;
				p.Ay[j] += im_r3 * ry;
				p.Az[j] += im_r3 * rz;
				p.Jx[j] += im_r3 * vx;
				p.Jy[j] += im_r3 * vy;
				p.Jz[j] += im_r3 * vz;
				p.Sx[j] += im_r3 * ax;
				p.Sy[j] += im_r3 * ay;
				p.Sz[j] += im_r3 * az;
				p.Cx[j] += im_r3 * jx;
				p.Cy[j] += im_r3 * jy;
				p.Cz[j] += im_r3 * jz;
			}
		}
	}
};

}	// namespace Snp_Crk
#endif


// ----------------------------------------------------------------------------


typedef struct snp_crk_data {
	union {
		real_tn m;
		real_t _m[SIMD];
	};
	union {
		real_tn e2;
		real_t _e2[SIMD];
	};
	union {
		real_tn rx;
		real_t _rx[SIMD];
	};
	union {
		real_tn ry;
		real_t _ry[SIMD];
	};
	union {
		real_tn rz;
		real_t _rz[SIMD];
	};
	union {
		real_tn vx;
		real_t _vx[SIMD];
	};
	union {
		real_tn vy;
		real_t _vy[SIMD];
	};
	union {
		real_tn vz;
		real_t _vz[SIMD];
	};
	union {
		real_tn ax;
		real_t _ax[SIMD];
	};
	union {
		real_tn ay;
		real_t _ay[SIMD];
	};
	union {
		real_tn az;
		real_t _az[SIMD];
	};
	union {
		real_tn jx;
		real_t _jx[SIMD];
	};
	union {
		real_tn jy;
		real_t _jy[SIMD];
	};
	union {
		real_tn jz;
		real_t _jz[SIMD];
	};
	union {
		real_tn Ax;
		real_t _Ax[SIMD];
	};
	union {
		real_tn Ay;
		real_t _Ay[SIMD];
	};
	union {
		real_tn Az;
		real_t _Az[SIMD];
	};
	union {
		real_tn Jx;
		real_t _Jx[SIMD];
	};
	union {
		real_tn Jy;
		real_t _Jy[SIMD];
	};
	union {
		real_tn Jz;
		real_t _Jz[SIMD];
	};
	union {
		real_tn Sx;
		real_t _Sx[SIMD];
	};
	union {
		real_tn Sy;
		real_t _Sy[SIMD];
	};
	union {
		real_tn Sz;
		real_t _Sz[SIMD];
	};
	union {
		real_tn Cx;
		real_t _Cx[SIMD];
	};
	union {
		real_tn Cy;
		real_t _Cy[SIMD];
	};
	union {
		real_tn Cz;
		real_t _Cz[SIMD];
	};
} Snp_Crk_Data;


static inline Snp_Crk_Data
snp_crk_kernel_core(Snp_Crk_Data ip, Snp_Crk_Data jp)
// flop count: 128
{
	real_tn ee = ip.e2 + jp.e2;
	real_tn rx = ip.rx - jp.rx;
	real_tn ry = ip.ry - jp.ry;
	real_tn rz = ip.rz - jp.rz;
	real_tn vx = ip.vx - jp.vx;
	real_tn vy = ip.vy - jp.vy;
	real_tn vz = ip.vz - jp.vz;
	real_tn ax = ip.ax - jp.ax;
	real_tn ay = ip.ay - jp.ay;
	real_tn az = ip.az - jp.az;
	real_tn jx = ip.jx - jp.jx;
	real_tn jy = ip.jy - jp.jy;
	real_tn jz = ip.jz - jp.jz;

	real_tn rr = ee;
	rr += rx * rx + ry * ry + rz * rz;

	real_tn inv_r3 = rsqrt(rr);
	inv_r3 = (rr > ee) ? (inv_r3):(0);
	real_tn inv_r2 = inv_r3 * inv_r3;
	inv_r3 *= inv_r2;
	inv_r2 *= -3;

	real_tn s1 = rx * vx + ry * vy + rz * vz;
	real_tn s2 = vx * vx + vy * vy + vz * vz;
	real_tn s3 = vx * ax + vy * ay + vz * az;
	s3 *= 3;
	s2 += rx * ax + ry * ay + rz * az;
	s3 += rx * jx + ry * jy + rz * jz;

	#define cq21 ((real_t)(5.0/3.0))
	#define cq31 ((real_t)(8.0/3.0))
	#define cq32 ((real_t)(7.0/3.0))

	const real_tn q1 = inv_r2 * (s1);
	const real_tn q2 = inv_r2 * (s2 + (cq21 * s1) * q1);
	const real_tn q3 = inv_r2 * (s3 + (cq31 * s2) * q1 + (cq32 * s1) * q2);

	const real_tn b3 = 3 * q1;
	const real_tn c3 = 3 * q2;
	const real_tn c2 = 2 * q1;

	jx += b3 * ax + c3 * vx + q3 * rx;
	jy += b3 * ay + c3 * vy + q3 * ry;
	jz += b3 * az + c3 * vz + q3 * rz;

	ax += c2 * vx + q2 * rx;
	ay += c2 * vy + q2 * ry;
	az += c2 * vz + q2 * rz;

	vx += q1 * rx;
	vy += q1 * ry;
	vz += q1 * rz;

	real_tn jm_r3 = jp.m * inv_r3;

	ip.Ax -= jm_r3 * rx;
	ip.Ay -= jm_r3 * ry;
	ip.Az -= jm_r3 * rz;
	ip.Jx -= jm_r3 * vx;
	ip.Jy -= jm_r3 * vy;
	ip.Jz -= jm_r3 * vz;
	ip.Sx -= jm_r3 * ax;
	ip.Sy -= jm_r3 * ay;
	ip.Sz -= jm_r3 * az;
	ip.Cx -= jm_r3 * jx;
	ip.Cy -= jm_r3 * jy;
	ip.Cz -= jm_r3 * jz;
	return ip;
}


#endif	// __SNP_CRK_KERNEL_COMMON_H__
