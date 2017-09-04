#ifndef __ACC_KERNEL_COMMON_H__
#define __ACC_KERNEL_COMMON_H__


#include "common.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
namespace Acc {

template<size_t TILE>
struct Acc_Data_SoA {
	real_t __ALIGNED__ m[TILE];
	real_t __ALIGNED__ e2[TILE];
	real_t __ALIGNED__ rx[TILE];
	real_t __ALIGNED__ ry[TILE];
	real_t __ALIGNED__ rz[TILE];
	real_t __ALIGNED__ ax[TILE];
	real_t __ALIGNED__ ay[TILE];
	real_t __ALIGNED__ az[TILE];
};

template<size_t TILE>
auto setup(
	const uint_t n,
	const real_t __m[],
	const real_t __e2[],
	const real_t __rdot[])
{
	auto ntiles = (n + TILE - 1) / TILE;
	vector<Acc_Data_SoA<TILE>> part(ntiles);
	for (size_t k = 0; k < n; ++k) {
		auto kk = k%TILE;
		auto& p = part[k/TILE];
		p.m[kk] = __m[k];
		p.e2[kk] = __e2[k];
		p.rx[kk] = __rdot[(0*NDIM+0)*n + k];
		p.ry[kk] = __rdot[(0*NDIM+1)*n + k];
		p.rz[kk] = __rdot[(0*NDIM+2)*n + k];
	}
	return part;
}

template<size_t TILE, typename PART>
void commit(const uint_t n, const PART& part, real_t __adot[])
{
	for (size_t k = 0; k < n; ++k) {
		auto kk = k%TILE;
		auto& p = part[k/TILE];
		__adot[(0*NDIM+0)*n + k] += p.ax[kk];
		__adot[(0*NDIM+1)*n + k] += p.ay[kk];
		__adot[(0*NDIM+2)*n + k] += p.az[kk];
	}
}

template<size_t TILE>
struct P2P_acc_kernel_core {
	template<typename IP, typename JP>
	void operator()(IP&& ip, JP&& jp) {
		// flop count: 28
		for (size_t i = 0; i < TILE; ++i) {
			auto iax = ip.ax[i];
			auto iay = ip.ay[i];
			auto iaz = ip.az[i];
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto ee = ip.e2[i] + jp.e2[j];
				auto rx = ip.rx[i] - jp.rx[j];
				auto ry = ip.ry[i] - jp.ry[j];
				auto rz = ip.rz[i] - jp.rz[j];

				auto rr = ee;
				rr += rx * rx + ry * ry + rz * rz;

				auto inv_r3 = rsqrt(rr);
				inv_r3 *= inv_r3 * inv_r3;

				auto im_r3 = ip.m[i] * inv_r3;
				jp.ax[j] += im_r3 * rx;
				jp.ay[j] += im_r3 * ry;
				jp.az[j] += im_r3 * rz;

				auto jm_r3 = jp.m[j] * inv_r3;
				iax -= jm_r3 * rx;
				iay -= jm_r3 * ry;
				iaz -= jm_r3 * rz;
			}
			ip.ax[i] = iax;
			ip.ay[i] = iay;
			ip.az[i] = iaz;
		}
	}

	template<typename P>
	void operator()(P&& p) {
		// flop count: 21
		for (size_t i = 0; i < TILE; ++i) {
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto ee = p.e2[i] + p.e2[j];
				auto rx = p.rx[i] - p.rx[j];
				auto ry = p.ry[i] - p.ry[j];
				auto rz = p.rz[i] - p.rz[j];

				auto rr = ee;
				rr += rx * rx + ry * ry + rz * rz;

				auto inv_r3 = rsqrt(rr);
				inv_r3 = (rr > ee) ? (inv_r3):(0);
				inv_r3 *= inv_r3 * inv_r3;

				auto im_r3 = p.m[i] * inv_r3;

				p.ax[j] += im_r3 * rx;
				p.ay[j] += im_r3 * ry;
				p.az[j] += im_r3 * rz;
			}
		}
	}
};

}	// namespace Acc
#else


// ----------------------------------------------------------------------------


#define DEFINE_ACC_DATA(TILE)			\
typedef struct concat(acc_data, TILE) {	\
	union {								\
		real_tn m[TILE];				\
		real_t _m[TILE * SIMD];			\
	};									\
	union {								\
		real_tn e2[TILE];				\
		real_t _e2[TILE * SIMD];		\
	};									\
	union {								\
		real_tn rx[TILE];				\
		real_t _rx[TILE * SIMD];		\
	};									\
	union {								\
		real_tn ry[TILE];				\
		real_t _ry[TILE * SIMD];		\
	};									\
	union {								\
		real_tn rz[TILE];				\
		real_t _rz[TILE * SIMD];		\
	};									\
	union {								\
		real_tn ax[TILE];				\
		real_t _ax[TILE * SIMD];		\
	};									\
	union {								\
		real_tn ay[TILE];				\
		real_t _ay[TILE * SIMD];		\
	};									\
	union {								\
		real_tn az[TILE];				\
		real_t _az[TILE * SIMD];		\
	};									\
} concat(Acc_Data, TILE);				\

DEFINE_ACC_DATA(1)
#if WPT != 1
DEFINE_ACC_DATA(WPT)
#endif
#if WGSIZE != 1 && WGSIZE != WPT
DEFINE_ACC_DATA(WGSIZE)
#endif


#define DEFINE_LOAD_ACC_DATA(TILE)					\
static inline void									\
concat(load_Acc_Data, TILE)(						\
	concat(Acc_Data, TILE) *p,						\
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
		}											\
	}												\
}													\

DEFINE_LOAD_ACC_DATA(1)
#if WPT != 1
DEFINE_LOAD_ACC_DATA(WPT)
#endif


#define DEFINE_STORE_ACC_DATA(TILE)								\
static inline void												\
concat(store_Acc_Data, TILE)(									\
	concat(Acc_Data, TILE) *p,									\
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
			atomic_fadd(&(__adot+(0*NDIM+0)*n)[kk], p->_ax[k]);	\
			atomic_fadd(&(__adot+(0*NDIM+1)*n)[kk], p->_ay[k]);	\
			atomic_fadd(&(__adot+(0*NDIM+2)*n)[kk], p->_az[k]);	\
		}														\
	}															\
}																\

DEFINE_STORE_ACC_DATA(1)
#if WPT != 1
DEFINE_STORE_ACC_DATA(WPT)
#endif


static inline void
simd_shuff_Acc_Data(
	const uint_t k,
	concat(Acc_Data, WPT) *p)
{
	shuff(p->m[k], SIMD);
	shuff(p->e2[k], SIMD);
	shuff(p->rx[k], SIMD);
	shuff(p->ry[k], SIMD);
	shuff(p->rz[k], SIMD);
	shuff(p->ax[k], SIMD);
	shuff(p->ay[k], SIMD);
	shuff(p->az[k], SIMD);
}


#endif	// __cplusplus
#endif	// __ACC_KERNEL_COMMON_H__
