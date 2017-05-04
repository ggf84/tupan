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


#define DEFINE_SNP_CRK_DATA(TILE)			\
typedef struct concat(snp_crk_data, TILE) {	\
	union {									\
		real_tn m[TILE];					\
		real_t _m[TILE * SIMD];				\
	};										\
	union {									\
		real_tn e2[TILE];					\
		real_t _e2[TILE * SIMD];			\
	};										\
	union {									\
		real_tn rx[TILE];					\
		real_t _rx[TILE * SIMD];			\
	};										\
	union {									\
		real_tn ry[TILE];					\
		real_t _ry[TILE * SIMD];			\
	};										\
	union {									\
		real_tn rz[TILE];					\
		real_t _rz[TILE * SIMD];			\
	};										\
	union {									\
		real_tn vx[TILE];					\
		real_t _vx[TILE * SIMD];			\
	};										\
	union {									\
		real_tn vy[TILE];					\
		real_t _vy[TILE * SIMD];			\
	};										\
	union {									\
		real_tn vz[TILE];					\
		real_t _vz[TILE * SIMD];			\
	};										\
	union {									\
		real_tn ax[TILE];					\
		real_t _ax[TILE * SIMD];			\
	};										\
	union {									\
		real_tn ay[TILE];					\
		real_t _ay[TILE * SIMD];			\
	};										\
	union {									\
		real_tn az[TILE];					\
		real_t _az[TILE * SIMD];			\
	};										\
	union {									\
		real_tn jx[TILE];					\
		real_t _jx[TILE * SIMD];			\
	};										\
	union {									\
		real_tn jy[TILE];					\
		real_t _jy[TILE * SIMD];			\
	};										\
	union {									\
		real_tn jz[TILE];					\
		real_t _jz[TILE * SIMD];			\
	};										\
	union {									\
		real_tn Ax[TILE];					\
		real_t _Ax[TILE * SIMD];			\
	};										\
	union {									\
		real_tn Ay[TILE];					\
		real_t _Ay[TILE * SIMD];			\
	};										\
	union {									\
		real_tn Az[TILE];					\
		real_t _Az[TILE * SIMD];			\
	};										\
	union {									\
		real_tn Jx[TILE];					\
		real_t _Jx[TILE * SIMD];			\
	};										\
	union {									\
		real_tn Jy[TILE];					\
		real_t _Jy[TILE * SIMD];			\
	};										\
	union {									\
		real_tn Jz[TILE];					\
		real_t _Jz[TILE * SIMD];			\
	};										\
	union {									\
		real_tn Sx[TILE];					\
		real_t _Sx[TILE * SIMD];			\
	};										\
	union {									\
		real_tn Sy[TILE];					\
		real_t _Sy[TILE * SIMD];			\
	};										\
	union {									\
		real_tn Sz[TILE];					\
		real_t _Sz[TILE * SIMD];			\
	};										\
	union {									\
		real_tn Cx[TILE];					\
		real_t _Cx[TILE * SIMD];			\
	};										\
	union {									\
		real_tn Cy[TILE];					\
		real_t _Cy[TILE * SIMD];			\
	};										\
	union {									\
		real_tn Cz[TILE];					\
		real_t _Cz[TILE * SIMD];			\
	};										\
} concat(Snp_Crk_Data, TILE);				\

DEFINE_SNP_CRK_DATA(1)
#if WPT != 1
DEFINE_SNP_CRK_DATA(WPT)
#endif
#if NLANES != 1 && NLANES != WPT
DEFINE_SNP_CRK_DATA(NLANES)
#endif


#define DEFINE_LOAD_SNP_CRK_DATA(TILE)				\
static inline void									\
concat(load_Snp_Crk_Data, TILE)(					\
	concat(Snp_Crk_Data, TILE) *p,					\
	const uint_t base,								\
	const uint_t stride,							\
	const uint_t nitems,							\
	const uint_t n,									\
	global const real_t __m[],						\
	global const real_t __e2[],						\
	global const real_t __rdot[])					\
{													\
	for (uint_t k = 0, kk = base;					\
				k < TILE * nitems;					\
				k += 1, kk += stride) {				\
		if (kk < n) {								\
			p->_m[k] = __m[kk];						\
			p->_e2[k] = __e2[kk];					\
			p->_rx[k] = (__rdot+(0*NDIM+0)*n)[kk];	\
			p->_ry[k] = (__rdot+(0*NDIM+1)*n)[kk];	\
			p->_rz[k] = (__rdot+(0*NDIM+2)*n)[kk];	\
			p->_vx[k] = (__rdot+(1*NDIM+0)*n)[kk];	\
			p->_vy[k] = (__rdot+(1*NDIM+1)*n)[kk];	\
			p->_vz[k] = (__rdot+(1*NDIM+2)*n)[kk];	\
			p->_ax[k] = (__rdot+(2*NDIM+0)*n)[kk];	\
			p->_ay[k] = (__rdot+(2*NDIM+1)*n)[kk];	\
			p->_az[k] = (__rdot+(2*NDIM+2)*n)[kk];	\
			p->_jx[k] = (__rdot+(3*NDIM+0)*n)[kk];	\
			p->_jy[k] = (__rdot+(3*NDIM+1)*n)[kk];	\
			p->_jz[k] = (__rdot+(3*NDIM+2)*n)[kk];	\
		}											\
	}												\
}													\

DEFINE_LOAD_SNP_CRK_DATA(1)
#if WPT != 1
DEFINE_LOAD_SNP_CRK_DATA(WPT)
#endif


#define DEFINE_STORE_SNP_CRK_DATA(TILE)							\
static inline void												\
concat(store_Snp_Crk_Data, TILE)(								\
	concat(Snp_Crk_Data, TILE) *p,								\
	const uint_t base,											\
	const uint_t stride,										\
	const uint_t nitems,										\
	const uint_t n,												\
	global real_t __adot[])										\
{																\
	for (uint_t k = 0, kk = base;								\
				k < TILE * nitems;								\
				k += 1, kk += stride) {							\
		if (kk < n) {											\
			atomic_fadd(&(__adot+(0*NDIM+0)*n)[kk], p->_Ax[k]);	\
			atomic_fadd(&(__adot+(0*NDIM+1)*n)[kk], p->_Ay[k]);	\
			atomic_fadd(&(__adot+(0*NDIM+2)*n)[kk], p->_Az[k]);	\
			atomic_fadd(&(__adot+(1*NDIM+0)*n)[kk], p->_Jx[k]);	\
			atomic_fadd(&(__adot+(1*NDIM+1)*n)[kk], p->_Jy[k]);	\
			atomic_fadd(&(__adot+(1*NDIM+2)*n)[kk], p->_Jz[k]);	\
			atomic_fadd(&(__adot+(2*NDIM+0)*n)[kk], p->_Sx[k]);	\
			atomic_fadd(&(__adot+(2*NDIM+1)*n)[kk], p->_Sy[k]);	\
			atomic_fadd(&(__adot+(2*NDIM+2)*n)[kk], p->_Sz[k]);	\
			atomic_fadd(&(__adot+(3*NDIM+0)*n)[kk], p->_Cx[k]);	\
			atomic_fadd(&(__adot+(3*NDIM+1)*n)[kk], p->_Cy[k]);	\
			atomic_fadd(&(__adot+(3*NDIM+2)*n)[kk], p->_Cz[k]);	\
		}														\
	}															\
}																\

DEFINE_STORE_SNP_CRK_DATA(1)
#if WPT != 1
DEFINE_STORE_SNP_CRK_DATA(WPT)
#endif


static inline void
simd_shuff_Snp_Crk_Data(
	const uint_t k,
	concat(Snp_Crk_Data, WPT) *p)
{
	shuff(p->m[k], SIMD);
	shuff(p->e2[k], SIMD);
	shuff(p->rx[k], SIMD);
	shuff(p->ry[k], SIMD);
	shuff(p->rz[k], SIMD);
	shuff(p->vx[k], SIMD);
	shuff(p->vy[k], SIMD);
	shuff(p->vz[k], SIMD);
	shuff(p->ax[k], SIMD);
	shuff(p->ay[k], SIMD);
	shuff(p->az[k], SIMD);
	shuff(p->jx[k], SIMD);
	shuff(p->jy[k], SIMD);
	shuff(p->jz[k], SIMD);
	shuff(p->Ax[k], SIMD);
	shuff(p->Ay[k], SIMD);
	shuff(p->Az[k], SIMD);
	shuff(p->Jx[k], SIMD);
	shuff(p->Jy[k], SIMD);
	shuff(p->Jz[k], SIMD);
	shuff(p->Sx[k], SIMD);
	shuff(p->Sy[k], SIMD);
	shuff(p->Sz[k], SIMD);
	shuff(p->Cx[k], SIMD);
	shuff(p->Cy[k], SIMD);
	shuff(p->Cz[k], SIMD);
}


#endif	// __cplusplus
#endif	// __SNP_CRK_KERNEL_COMMON_H__
