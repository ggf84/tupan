#ifndef __SNP_CRK_KERNEL_COMMON_H__
#define __SNP_CRK_KERNEL_COMMON_H__


#include "common.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
namespace Snp_Crk {

template<size_t TILE>
struct Snp_Crk_Data_SoA {
	real_t __ALIGNED__ m[TILE];
	real_t __ALIGNED__ e2[TILE];
	real_t __ALIGNED__ rx[TILE];
	real_t __ALIGNED__ ry[TILE];
	real_t __ALIGNED__ rz[TILE];
	real_t __ALIGNED__ vx[TILE];
	real_t __ALIGNED__ vy[TILE];
	real_t __ALIGNED__ vz[TILE];
	real_t __ALIGNED__ ax[TILE];
	real_t __ALIGNED__ ay[TILE];
	real_t __ALIGNED__ az[TILE];
	real_t __ALIGNED__ jx[TILE];
	real_t __ALIGNED__ jy[TILE];
	real_t __ALIGNED__ jz[TILE];
	real_t __ALIGNED__ Ax[TILE];
	real_t __ALIGNED__ Ay[TILE];
	real_t __ALIGNED__ Az[TILE];
	real_t __ALIGNED__ Jx[TILE];
	real_t __ALIGNED__ Jy[TILE];
	real_t __ALIGNED__ Jz[TILE];
	real_t __ALIGNED__ Sx[TILE];
	real_t __ALIGNED__ Sy[TILE];
	real_t __ALIGNED__ Sz[TILE];
	real_t __ALIGNED__ Cx[TILE];
	real_t __ALIGNED__ Cy[TILE];
	real_t __ALIGNED__ Cz[TILE];
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
		for (size_t i = 0; i < TILE; ++i) {
			auto iAx = ip.Ax[i];
			auto iAy = ip.Ay[i];
			auto iAz = ip.Az[i];
			auto iJx = ip.Jx[i];
			auto iJy = ip.Jy[i];
			auto iJz = ip.Jz[i];
			auto iSx = ip.Sx[i];
			auto iSy = ip.Sy[i];
			auto iSz = ip.Sz[i];
			auto iCx = ip.Cx[i];
			auto iCy = ip.Cy[i];
			auto iCz = ip.Cz[i];
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto ee = ip.e2[i] + jp.e2[j];
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

				auto rr = ee;
				rr += rx * rx + ry * ry + rz * rz;

				auto inv_r3 = rsqrt(rr);
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

				auto im_r3 = ip.m[i] * inv_r3;
				jp.Ax[j] += im_r3 * rx;
				jp.Ay[j] += im_r3 * ry;
				jp.Az[j] += im_r3 * rz;
				jp.Jx[j] += im_r3 * vx;
				jp.Jy[j] += im_r3 * vy;
				jp.Jz[j] += im_r3 * vz;
				jp.Sx[j] += im_r3 * ax;
				jp.Sy[j] += im_r3 * ay;
				jp.Sz[j] += im_r3 * az;
				jp.Cx[j] += im_r3 * jx;
				jp.Cy[j] += im_r3 * jy;
				jp.Cz[j] += im_r3 * jz;

				auto jm_r3 = jp.m[j] * inv_r3;
				iAx -= jm_r3 * rx;
				iAy -= jm_r3 * ry;
				iAz -= jm_r3 * rz;
				iJx -= jm_r3 * vx;
				iJy -= jm_r3 * vy;
				iJz -= jm_r3 * vz;
				iSx -= jm_r3 * ax;
				iSy -= jm_r3 * ay;
				iSz -= jm_r3 * az;
				iCx -= jm_r3 * jx;
				iCy -= jm_r3 * jy;
				iCz -= jm_r3 * jz;
			}
			ip.Ax[i] = iAx;
			ip.Ay[i] = iAy;
			ip.Az[i] = iAz;
			ip.Jx[i] = iJx;
			ip.Jy[i] = iJy;
			ip.Jz[i] = iJz;
			ip.Sx[i] = iSx;
			ip.Sy[i] = iSy;
			ip.Sz[i] = iSz;
			ip.Cx[i] = iCx;
			ip.Cy[i] = iCy;
			ip.Cz[i] = iCz;
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
#else


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


typedef struct snp_crk_data_soa {
	union {
		real_tn m[LMSIZE];
		real_t _m[LMSIZE * SIMD];
	};
	union {
		real_tn e2[LMSIZE];
		real_t _e2[LMSIZE * SIMD];
	};
	union {
		real_tn rx[LMSIZE];
		real_t _rx[LMSIZE * SIMD];
	};
	union {
		real_tn ry[LMSIZE];
		real_t _ry[LMSIZE * SIMD];
	};
	union {
		real_tn rz[LMSIZE];
		real_t _rz[LMSIZE * SIMD];
	};
	union {
		real_tn vx[LMSIZE];
		real_t _vx[LMSIZE * SIMD];
	};
	union {
		real_tn vy[LMSIZE];
		real_t _vy[LMSIZE * SIMD];
	};
	union {
		real_tn vz[LMSIZE];
		real_t _vz[LMSIZE * SIMD];
	};
	union {
		real_tn ax[LMSIZE];
		real_t _ax[LMSIZE * SIMD];
	};
	union {
		real_tn ay[LMSIZE];
		real_t _ay[LMSIZE * SIMD];
	};
	union {
		real_tn az[LMSIZE];
		real_t _az[LMSIZE * SIMD];
	};
	union {
		real_tn jx[LMSIZE];
		real_t _jx[LMSIZE * SIMD];
	};
	union {
		real_tn jy[LMSIZE];
		real_t _jy[LMSIZE * SIMD];
	};
	union {
		real_tn jz[LMSIZE];
		real_t _jz[LMSIZE * SIMD];
	};
	union {
		real_tn Ax[LMSIZE];
		real_t _Ax[LMSIZE * SIMD];
	};
	union {
		real_tn Ay[LMSIZE];
		real_t _Ay[LMSIZE * SIMD];
	};
	union {
		real_tn Az[LMSIZE];
		real_t _Az[LMSIZE * SIMD];
	};
	union {
		real_tn Jx[LMSIZE];
		real_t _Jx[LMSIZE * SIMD];
	};
	union {
		real_tn Jy[LMSIZE];
		real_t _Jy[LMSIZE * SIMD];
	};
	union {
		real_tn Jz[LMSIZE];
		real_t _Jz[LMSIZE * SIMD];
	};
	union {
		real_tn Sx[LMSIZE];
		real_t _Sx[LMSIZE * SIMD];
	};
	union {
		real_tn Sy[LMSIZE];
		real_t _Sy[LMSIZE * SIMD];
	};
	union {
		real_tn Sz[LMSIZE];
		real_t _Sz[LMSIZE * SIMD];
	};
	union {
		real_tn Cx[LMSIZE];
		real_t _Cx[LMSIZE * SIMD];
	};
	union {
		real_tn Cy[LMSIZE];
		real_t _Cy[LMSIZE * SIMD];
	};
	union {
		real_tn Cz[LMSIZE];
		real_t _Cz[LMSIZE * SIMD];
	};
} Snp_Crk_Data_SoA;


static inline void
read_Snp_Crk_Data(
	uint_t base,
	uint_t lid,
	Snp_Crk_Data *p,
	uint_t n,
	global const real_t __m[],
	global const real_t __e2[],
	global const real_t __rdot[])
{
	for (uint_t k = 0, kk = base + lid;
				k < SIMD;
				k += 1, kk += WGSIZE) {
		if (kk < n) {
			p->_m[k] = __m[kk];
			p->_e2[k] = __e2[kk];
			p->_rx[k] = (__rdot+(0*NDIM+0)*n)[kk];
			p->_ry[k] = (__rdot+(0*NDIM+1)*n)[kk];
			p->_rz[k] = (__rdot+(0*NDIM+2)*n)[kk];
			p->_vx[k] = (__rdot+(1*NDIM+0)*n)[kk];
			p->_vy[k] = (__rdot+(1*NDIM+1)*n)[kk];
			p->_vz[k] = (__rdot+(1*NDIM+2)*n)[kk];
			p->_ax[k] = (__rdot+(2*NDIM+0)*n)[kk];
			p->_ay[k] = (__rdot+(2*NDIM+1)*n)[kk];
			p->_az[k] = (__rdot+(2*NDIM+2)*n)[kk];
			p->_jx[k] = (__rdot+(3*NDIM+0)*n)[kk];
			p->_jy[k] = (__rdot+(3*NDIM+1)*n)[kk];
			p->_jz[k] = (__rdot+(3*NDIM+2)*n)[kk];
		}
	}
}


static inline void
simd_shuff_Snp_Crk_Data(Snp_Crk_Data *p)
{
	shuff(p->m, SIMD);
	shuff(p->e2, SIMD);
	shuff(p->rx, SIMD);
	shuff(p->ry, SIMD);
	shuff(p->rz, SIMD);
	shuff(p->vx, SIMD);
	shuff(p->vy, SIMD);
	shuff(p->vz, SIMD);
	shuff(p->ax, SIMD);
	shuff(p->ay, SIMD);
	shuff(p->az, SIMD);
	shuff(p->jx, SIMD);
	shuff(p->jy, SIMD);
	shuff(p->jz, SIMD);
	shuff(p->Ax, SIMD);
	shuff(p->Ay, SIMD);
	shuff(p->Az, SIMD);
	shuff(p->Jx, SIMD);
	shuff(p->Jy, SIMD);
	shuff(p->Jz, SIMD);
	shuff(p->Sx, SIMD);
	shuff(p->Sy, SIMD);
	shuff(p->Sz, SIMD);
	shuff(p->Cx, SIMD);
	shuff(p->Cy, SIMD);
	shuff(p->Cz, SIMD);
}


#endif	// __cplusplus
#endif	// __SNP_CRK_KERNEL_COMMON_H__
