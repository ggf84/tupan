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
	const real_t __rdot[])
{
	auto ntiles = (n + TILE - 1) / TILE;
	vector<PNAcc_Data_SoA<TILE>> part(ntiles);
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
	}
	return part;
}

template<size_t TILE, typename PART>
void commit(const uint_t n, const PART& part, real_t __pnacc[])
{
	for (size_t k = 0; k < n; ++k) {
		auto kk = k%TILE;
		auto& p = part[k/TILE];
		__pnacc[(0*NDIM+0)*n + k] = p.pnax[kk];
		__pnacc[(0*NDIM+1)*n + k] = p.pnay[kk];
		__pnacc[(0*NDIM+2)*n + k] = p.pnaz[kk];
	}
}

template<size_t TILE>
struct P2P_pnacc_kernel_core {
	const CLIGHT clight;
	P2P_pnacc_kernel_core(const CLIGHT& clight) : clight(clight) {}

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

				auto order = clight.order;
				auto inv_c = clight.inv1;

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

				auto order = clight.order;
				auto inv_c = clight.inv1;

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


typedef struct pnacc_data {
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
		real_tn pnax;
		real_t _pnax[SIMD];
	};
	union {
		real_tn pnay;
		real_t _pnay[SIMD];
	};
	union {
		real_tn pnaz;
		real_t _pnaz[SIMD];
	};
} PNAcc_Data;


typedef struct pnacc_data_soa {
	union {
		real_tn m[NLANES];
		real_t _m[NLANES * SIMD];
	};
	union {
		real_tn e2[NLANES];
		real_t _e2[NLANES * SIMD];
	};
	union {
		real_tn rx[NLANES];
		real_t _rx[NLANES * SIMD];
	};
	union {
		real_tn ry[NLANES];
		real_t _ry[NLANES * SIMD];
	};
	union {
		real_tn rz[NLANES];
		real_t _rz[NLANES * SIMD];
	};
	union {
		real_tn vx[NLANES];
		real_t _vx[NLANES * SIMD];
	};
	union {
		real_tn vy[NLANES];
		real_t _vy[NLANES * SIMD];
	};
	union {
		real_tn vz[NLANES];
		real_t _vz[NLANES * SIMD];
	};
	union {
		real_tn pnax[NLANES];
		real_t _pnax[NLANES * SIMD];
	};
	union {
		real_tn pnay[NLANES];
		real_t _pnay[NLANES * SIMD];
	};
	union {
		real_tn pnaz[NLANES];
		real_t _pnaz[NLANES * SIMD];
	};
} PNAcc_Data_SoA;


static inline void
read_PNAcc_Data(
	uint_t base,
	uint_t lid,
	PNAcc_Data *p,
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
		}
	}
}


static inline void
simd_shuff_PNAcc_Data(PNAcc_Data *p)
{
	shuff(p->m, SIMD);
	shuff(p->e2, SIMD);
	shuff(p->rx, SIMD);
	shuff(p->ry, SIMD);
	shuff(p->rz, SIMD);
	shuff(p->vx, SIMD);
	shuff(p->vy, SIMD);
	shuff(p->vz, SIMD);
	shuff(p->pnax, SIMD);
	shuff(p->pnay, SIMD);
	shuff(p->pnaz, SIMD);
}


#endif	// __cplusplus
#endif	// __PNACC_KERNEL_COMMON_H__
