#ifndef __PNACC_KERNEL_COMMON_H__
#define __PNACC_KERNEL_COMMON_H__


#include "common.h"
#include "pn_terms.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
namespace PNAcc {

template<size_t TILE>
struct PNAcc_Data_SoA {
	real_t __ALIGNED__ m[TILE];
	real_t __ALIGNED__ e2[TILE];
	real_t __ALIGNED__ rx[TILE];
	real_t __ALIGNED__ ry[TILE];
	real_t __ALIGNED__ rz[TILE];
	real_t __ALIGNED__ vx[TILE];
	real_t __ALIGNED__ vy[TILE];
	real_t __ALIGNED__ vz[TILE];
	real_t __ALIGNED__ pnax[TILE];
	real_t __ALIGNED__ pnay[TILE];
	real_t __ALIGNED__ pnaz[TILE];
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
	vector<PNAcc_Data_SoA<TILE>> part(ntiles);
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
void commit(const uint_t n, const PART& part, real_t __pnacc[])
{
	for (size_t k = 0; k < n; ++k) {
		auto kk = k%TILE;
		auto& p = part[k/TILE];
		__pnacc[0*n + k] += p.pnax[kk];
		__pnacc[1*n + k] += p.pnay[kk];
		__pnacc[2*n + k] += p.pnaz[kk];
	}
}

template<size_t TILE>
struct P2P_pnacc_kernel_core {
	const uint_t order;
	const real_t clight;
	P2P_pnacc_kernel_core(const uint_t& order, const real_t& clight) :
		order(order), clight(clight)
		{}

	template<typename IP, typename JP>
	void operator()(IP&& ip, JP&& jp) {
		// flop count: 45 + ???
		for (size_t i = 0; i < TILE; ++i) {
			auto ipnax = ip.pnax[i];
			auto ipnay = ip.pnay[i];
			auto ipnaz = ip.pnaz[i];
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

				auto inv_r = rsqrt(rr);
				auto inv_r2 = inv_r * inv_r;

				auto im = ip.m[i];
				auto im2 = im * im;
				auto im_r = im * inv_r;
				auto iv2 = ip.vx[i] * ip.vx[i]
						 + ip.vy[i] * ip.vy[i]
						 + ip.vz[i] * ip.vz[i];
				auto iv4 = iv2 * iv2;
				auto niv = rx * ip.vx[i]
						 + ry * ip.vy[i]
						 + rz * ip.vz[i];
				niv *= inv_r;
				auto niv2 = niv * niv;

				auto jm = jp.m[j];
				auto jm2 = jm * jm;
				auto jm_r = jm * inv_r;
				auto jv2 = jp.vx[j] * jp.vx[j]
						 + jp.vy[j] * jp.vy[j]
						 + jp.vz[j] * jp.vz[j];
				auto jv4 = jv2 * jv2;
				auto njv = rx * jp.vx[j]
						 + ry * jp.vy[j]
						 + rz * jp.vz[j];
				njv *= inv_r;
				auto njv2 = njv * njv;

				auto imjm = im * jm;
				auto vv = vx * vx
						+ vy * vy
						+ vz * vz;
				auto ivjv = ip.vx[i] * jp.vx[j]
						  + ip.vy[i] * jp.vy[j]
						  + ip.vz[i] * jp.vz[j];
				auto nv = rx * vx
						+ ry * vy
						+ rz * vz;
				nv *= inv_r;
				auto nvnv = nv * nv;
				auto nivnjv = niv * njv;

				auto inv_c = 1 / clight;

				auto ipnA = pnterms_A(jm, jm2, jm_r, jv2, jv4, -njv, njv2,
									  im, im2, im_r, iv2, iv4, -niv, niv2,
									  imjm, inv_r, inv_r2, vv, ivjv,
									  nv, nvnv, nivnjv, order, inv_c);
				auto ipnB = pnterms_B(jm, jm2, jm_r, jv2, jv4, -njv, njv2,
									  im, im2, im_r, iv2, iv4, -niv, niv2,
									  imjm, inv_r, inv_r2, vv, ivjv,
									  nv, nvnv, nivnjv, order, inv_c);
				jp.pnax[j] -= ipnA * rx + ipnB * vx;
				jp.pnay[j] -= ipnA * ry + ipnB * vy;
				jp.pnaz[j] -= ipnA * rz + ipnB * vz;

				auto jpnA = pnterms_A(im, im2, im_r, iv2, iv4, +niv, niv2,
									  jm, jm2, jm_r, jv2, jv4, +njv, njv2,
									  imjm, inv_r, inv_r2, vv, ivjv,
									  nv, nvnv, nivnjv, order, inv_c);
				auto jpnB = pnterms_B(im, im2, im_r, iv2, iv4, +niv, niv2,
									  jm, jm2, jm_r, jv2, jv4, +njv, njv2,
									  imjm, inv_r, inv_r2, vv, ivjv,
									  nv, nvnv, nivnjv, order, inv_c);
				ipnax += jpnA * rx + jpnB * vx;
				ipnay += jpnA * ry + jpnB * vy;
				ipnaz += jpnA * rz + jpnB * vz;
			}
			ip.pnax[i] = ipnax;
			ip.pnay[i] = ipnay;
			ip.pnaz[i] = ipnaz;
		}
	}

	template<typename P>
	void operator()(P&& p) {
		// flop count: 33 + ???
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

				auto inv_r = rsqrt(rr);
				inv_r = (rr > ee) ? (inv_r):(0);
				auto inv_r2 = inv_r * inv_r;

				auto im = p.m[i];
				auto im2 = im * im;
				auto im_r = im * inv_r;
				auto iv2 = p.vx[i] * p.vx[i]
						 + p.vy[i] * p.vy[i]
						 + p.vz[i] * p.vz[i];
				auto iv4 = iv2 * iv2;
				auto niv = rx * p.vx[i]
						 + ry * p.vy[i]
						 + rz * p.vz[i];
				niv *= inv_r;
				auto niv2 = niv * niv;

				auto jm = p.m[j];
				auto jm2 = jm * jm;
				auto jm_r = jm * inv_r;
				auto jv2 = p.vx[j] * p.vx[j]
						 + p.vy[j] * p.vy[j]
						 + p.vz[j] * p.vz[j];
				auto jv4 = jv2 * jv2;
				auto njv = rx * p.vx[j]
						 + ry * p.vy[j]
						 + rz * p.vz[j];
				njv *= inv_r;
				auto njv2 = njv * njv;

				auto imjm = im * jm;
				auto vv = vx * vx
						+ vy * vy
						+ vz * vz;
				auto ivjv = p.vx[i] * p.vx[j]
						  + p.vy[i] * p.vy[j]
						  + p.vz[i] * p.vz[j];
				auto nv = rx * vx
						+ ry * vy
						+ rz * vz;
				nv *= inv_r;
				auto nvnv = nv * nv;
				auto nivnjv = niv * njv;

				auto inv_c = 1 / clight;

				auto ipnA = pnterms_A(jm, jm2, jm_r, jv2, jv4, -njv, njv2,
									  im, im2, im_r, iv2, iv4, -niv, niv2,
									  imjm, inv_r, inv_r2, vv, ivjv,
									  nv, nvnv, nivnjv, order, inv_c);
				auto ipnB = pnterms_B(jm, jm2, jm_r, jv2, jv4, -njv, njv2,
									  im, im2, im_r, iv2, iv4, -niv, niv2,
									  imjm, inv_r, inv_r2, vv, ivjv,
									  nv, nvnv, nivnjv, order, inv_c);
				p.pnax[j] -= ipnA * rx + ipnB * vx;
				p.pnay[j] -= ipnA * ry + ipnB * vy;
				p.pnaz[j] -= ipnA * rz + ipnB * vz;
			}
		}
	}
};

}	// namespace PNAcc
#else


// ----------------------------------------------------------------------------


#define DEFINE_PNACC_DATA(TILE)				\
typedef struct concat(pnacc_data, TILE) {	\
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
		real_tn pnax[TILE];					\
		real_t _pnax[TILE * SIMD];			\
	};										\
	union {									\
		real_tn pnay[TILE];					\
		real_t _pnay[TILE * SIMD];			\
	};										\
	union {									\
		real_tn pnaz[TILE];					\
		real_t _pnaz[TILE * SIMD];			\
	};										\
} concat(PNAcc_Data, TILE);					\

DEFINE_PNACC_DATA(1)
#if WPT != 1
DEFINE_PNACC_DATA(WPT)
#endif
#if WGSIZE != 1 && WGSIZE != WPT
DEFINE_PNACC_DATA(WGSIZE)
#endif


#define DEFINE_LOAD_PNACC_DATA(TILE)				\
static inline void									\
concat(load_PNAcc_Data, TILE)(						\
	concat(PNAcc_Data, TILE) *p,					\
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

DEFINE_LOAD_PNACC_DATA(1)
#if WPT != 1
DEFINE_LOAD_PNACC_DATA(WPT)
#endif


#define DEFINE_STORE_PNACC_DATA(TILE)								\
static inline void													\
concat(store_PNAcc_Data, TILE)(										\
	concat(PNAcc_Data, TILE) *p,									\
	const uint_t base,												\
	const uint_t stride,											\
	const uint_t nitems,											\
	const uint_t n,													\
	global real_t __pnacc[])										\
{																	\
	for (uint_t k = 0, kk = base;									\
				k < TILE * nitems;									\
				k += 1, kk += stride) {								\
		if (kk < n) {												\
			atomic_fadd(&(__pnacc+0*n)[kk], -p->_pnax[k]);			\
			atomic_fadd(&(__pnacc+1*n)[kk], -p->_pnay[k]);			\
			atomic_fadd(&(__pnacc+2*n)[kk], -p->_pnaz[k]);			\
		}															\
	}																\
}																	\

DEFINE_STORE_PNACC_DATA(1)
#if WPT != 1
DEFINE_STORE_PNACC_DATA(WPT)
#endif


static inline void
simd_shuff_PNAcc_Data(
	const uint_t k,
	concat(PNAcc_Data, WPT) *p)
{
	shuff(p->m[k], SIMD);
	shuff(p->e2[k], SIMD);
	shuff(p->rx[k], SIMD);
	shuff(p->ry[k], SIMD);
	shuff(p->rz[k], SIMD);
	shuff(p->vx[k], SIMD);
	shuff(p->vy[k], SIMD);
	shuff(p->vz[k], SIMD);
	shuff(p->pnax[k], SIMD);
	shuff(p->pnay[k], SIMD);
	shuff(p->pnaz[k], SIMD);
}


#endif	// __cplusplus
#endif	// __PNACC_KERNEL_COMMON_H__
