#ifndef __ACC_JRK_KERNEL_COMMON_H__
#define __ACC_JRK_KERNEL_COMMON_H__


#include "common.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
namespace Acc_Jrk {

template<size_t TILE>
struct Acc_Jrk_Data_SoA {
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
};

template<size_t TILE>
auto setup(
	const uint_t n,
	const real_t __m[],
	const real_t __e2[],
	const real_t __pos[],
	const real_t __vel[])
{
	auto ntiles = (n + TILE - 1) / TILE;
	vector<Acc_Jrk_Data_SoA<TILE>> part(ntiles);
	for (size_t k = 0; k < n; ++k) {
		auto kk = k%TILE;
		auto& p = part[k/TILE];
		p.m[kk] = __m[k];
		p.e2[kk] = __e2[k];
		p.rx[kk] = __pos[0*n + k];
		p.ry[kk] = __pos[1*n + k];
		p.rz[kk] = __pos[2*n + k];
		p.vx[kk] = __vel[0*n + k];
		p.vy[kk] = __vel[1*n + k];
		p.vz[kk] = __vel[2*n + k];
	}
	return part;
}

template<size_t TILE, typename PART>
void commit(const uint_t n, const PART& part,
			real_t __acc[], real_t __jrk[])
{
	for (size_t k = 0; k < n; ++k) {
		auto kk = k%TILE;
		auto& p = part[k/TILE];
		__acc[0*n + k] += p.ax[kk];
		__acc[1*n + k] += p.ay[kk];
		__acc[2*n + k] += p.az[kk];
		__jrk[0*n + k] += p.jx[kk];
		__jrk[1*n + k] += p.jy[kk];
		__jrk[2*n + k] += p.jz[kk];
	}
}

template<size_t TILE>
struct P2P_acc_jrk_kernel_core {
	template<typename IP, typename JP>
	void operator()(IP&& ip, JP&& jp) {
		// flop count: 56
		for (size_t i = 0; i < TILE; ++i) {
			auto iax = ip.ax[i];
			auto iay = ip.ay[i];
			auto iaz = ip.az[i];
			auto ijx = ip.jx[i];
			auto ijy = ip.jy[i];
			auto ijz = ip.jz[i];
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto ee = ip.e2[i] + jp.e2[j];
				auto rx = ip.rx[i] - jp.rx[j];
				auto ry = ip.ry[i] - jp.ry[j];
				auto rz = ip.rz[i] - jp.rz[j];
				auto vx = ip.vx[i] - jp.vx[j];
				auto vy = ip.vy[i] - jp.vy[j];
				auto vz = ip.vz[i] - jp.vz[j];

				auto rr = ee;
				rr += rx * rx + ry * ry + rz * rz;

				auto inv_r3 = rsqrt(rr);
				auto inv_r2 = inv_r3 * inv_r3;
				inv_r3 *= inv_r2;
				inv_r2 *= -3;

				auto s1 = rx * vx + ry * vy + rz * vz;

				auto q1 = inv_r2 * (s1);
				vx += q1 * rx;
				vy += q1 * ry;
				vz += q1 * rz;

				auto im_r3 = ip.m[i] * inv_r3;
				jp.ax[j] += im_r3 * rx;
				jp.ay[j] += im_r3 * ry;
				jp.az[j] += im_r3 * rz;
				jp.jx[j] += im_r3 * vx;
				jp.jy[j] += im_r3 * vy;
				jp.jz[j] += im_r3 * vz;

				auto jm_r3 = jp.m[j] * inv_r3;
				iax -= jm_r3 * rx;
				iay -= jm_r3 * ry;
				iaz -= jm_r3 * rz;
				ijx -= jm_r3 * vx;
				ijy -= jm_r3 * vy;
				ijz -= jm_r3 * vz;
			}
			ip.ax[i] = iax;
			ip.ay[i] = iay;
			ip.az[i] = iaz;
			ip.jx[i] = ijx;
			ip.jy[i] = ijy;
			ip.jz[i] = ijz;
		}
	}

	template<typename P>
	void operator()(P&& p) {
		// flop count: 43
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

				auto rr = ee;
				rr += rx * rx + ry * ry + rz * rz;

				auto inv_r3 = rsqrt(rr);
				inv_r3 = (rr > ee) ? (inv_r3):(0);
				auto inv_r2 = inv_r3 * inv_r3;
				inv_r3 *= inv_r2;
				inv_r2 *= -3;

				auto s1 = rx * vx + ry * vy + rz * vz;

				auto q1 = inv_r2 * (s1);
				vx += q1 * rx;
				vy += q1 * ry;
				vz += q1 * rz;

				auto im_r3 = p.m[i] * inv_r3;

				p.ax[j] += im_r3 * rx;
				p.ay[j] += im_r3 * ry;
				p.az[j] += im_r3 * rz;
				p.jx[j] += im_r3 * vx;
				p.jy[j] += im_r3 * vy;
				p.jz[j] += im_r3 * vz;
			}
		}
	}
};

}	// namespace Acc_Jrk
#else


// ----------------------------------------------------------------------------


#define DEFINE_ACC_JRK_DATA(TILE)			\
typedef struct concat(acc_jrk_data, TILE) {	\
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
} concat(Acc_Jrk_Data, TILE);				\

DEFINE_ACC_JRK_DATA(1)
#if WPT != 1
DEFINE_ACC_JRK_DATA(WPT)
#endif
#if WGSIZE != 1 && WGSIZE != WPT
DEFINE_ACC_JRK_DATA(WGSIZE)
#endif


#define DEFINE_LOAD_ACC_JRK_DATA(TILE)				\
static inline void									\
concat(load_Acc_Jrk_Data, TILE)(					\
	concat(Acc_Jrk_Data, TILE) *p,					\
	const uint_t base,								\
	const uint_t stride,							\
	const uint_t nitems,							\
	const uint_t n,									\
	global const real_t __m[],						\
	global const real_t __e2[],						\
	global const real_t __pos[],					\
	global const real_t __vel[])					\
{													\
	for (uint_t k = 0, kk = base;					\
				k < TILE * nitems;					\
				k += 1, kk += stride) {				\
		if (kk < n) {								\
			p->_m[k] = __m[kk];						\
			p->_e2[k] = __e2[kk];					\
			p->_rx[k] = (__pos+0*n)[kk];			\
			p->_ry[k] = (__pos+1*n)[kk];			\
			p->_rz[k] = (__pos+2*n)[kk];			\
			p->_vx[k] = (__vel+0*n)[kk];			\
			p->_vy[k] = (__vel+1*n)[kk];			\
			p->_vz[k] = (__vel+2*n)[kk];			\
		}											\
	}												\
}													\

DEFINE_LOAD_ACC_JRK_DATA(1)
#if WPT != 1
DEFINE_LOAD_ACC_JRK_DATA(WPT)
#endif


#define DEFINE_STORE_ACC_JRK_DATA(TILE)							\
static inline void												\
concat(store_Acc_Jrk_Data, TILE)(								\
	concat(Acc_Jrk_Data, TILE) *p,								\
	const uint_t base,											\
	const uint_t stride,										\
	const uint_t nitems,										\
	const uint_t n,												\
	global real_t __acc[],										\
	global real_t __jrk[])										\
{																\
	for (uint_t k = 0, kk = base;								\
				k < TILE * nitems;								\
				k += 1, kk += stride) {							\
		if (kk < n) {											\
			atomic_fadd(&(__acc+0*n)[kk], p->_ax[k]);			\
			atomic_fadd(&(__acc+1*n)[kk], p->_ay[k]);			\
			atomic_fadd(&(__acc+2*n)[kk], p->_az[k]);			\
			atomic_fadd(&(__jrk+0*n)[kk], p->_jx[k]);			\
			atomic_fadd(&(__jrk+1*n)[kk], p->_jy[k]);			\
			atomic_fadd(&(__jrk+2*n)[kk], p->_jz[k]);			\
		}														\
	}															\
}																\

DEFINE_STORE_ACC_JRK_DATA(1)
#if WPT != 1
DEFINE_STORE_ACC_JRK_DATA(WPT)
#endif


static inline void
simd_shuff_Acc_Jrk_Data(
	const uint_t k,
	concat(Acc_Jrk_Data, WPT) *p)
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
}


#endif	// __cplusplus
#endif	// __ACC_JRK_KERNEL_COMMON_H__
