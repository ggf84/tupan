#ifndef __PNACC_KERNEL_COMMON_H__
#define __PNACC_KERNEL_COMMON_H__


#include "common.h"
#include "pn_terms.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
namespace PNAcc {

template<size_t TILE>
struct PNAcc_Data_SoA {
	real_t m[TILE];
	real_t e2[TILE];
	real_t rx[TILE];
	real_t ry[TILE];
	real_t rz[TILE];
	real_t vx[TILE];
	real_t vy[TILE];
	real_t vz[TILE];
	real_t pnax[TILE];
	real_t pnay[TILE];
	real_t pnaz[TILE];
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
		decltype(jp.m) rx, ry, rz, vx, vy, vz, inv_r1;
		decltype(jp.m) ipnA, ipnB, jpnA, jpnB;
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

				auto rr = ee;
				rr += rx[j] * rx[j] + ry[j] * ry[j] + rz[j] * rz[j];

				inv_r1[j] = rsqrt(rr);
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto inv_r = inv_r1[j];
				auto inv_r2 = inv_r * inv_r;

				auto im = ip.m[i];
				auto im2 = im * im;
				auto im_r = im * inv_r;
				auto iv2 = ip.vx[i] * ip.vx[i]
						 + ip.vy[i] * ip.vy[i]
						 + ip.vz[i] * ip.vz[i];
				auto iv4 = iv2 * iv2;
				auto niv = rx[j] * ip.vx[i]
						 + ry[j] * ip.vy[i]
						 + rz[j] * ip.vz[i];
				niv *= inv_r;
				auto niv2 = niv * niv;

				auto jm = jp.m[j];
				auto jm2 = jm * jm;
				auto jm_r = jm * inv_r;
				auto jv2 = jp.vx[j] * jp.vx[j]
						 + jp.vy[j] * jp.vy[j]
						 + jp.vz[j] * jp.vz[j];
				auto jv4 = jv2 * jv2;
				auto njv = rx[j] * jp.vx[j]
						 + ry[j] * jp.vy[j]
						 + rz[j] * jp.vz[j];
				njv *= inv_r;
				auto njv2 = njv * njv;

				auto imjm = im * jm;
				auto vv = vx[j] * vx[j]
						+ vy[j] * vy[j]
						+ vz[j] * vz[j];
				auto ivjv = ip.vx[i] * jp.vx[j]
						  + ip.vy[i] * jp.vy[j]
						  + ip.vz[i] * jp.vz[j];
				auto nv = rx[j] * vx[j]
						+ ry[j] * vy[j]
						+ rz[j] * vz[j];
				nv *= inv_r;
				auto nvnv = nv * nv;
				auto nivnjv = niv * njv;

				auto order = clight.order;
				auto inv_c = clight.inv1;

				ipnA[j] = pnterms_A(jm, jm2, jm_r, jv2, jv4, -njv, njv2,
									im, im2, im_r, iv2, iv4, -niv, niv2,
									imjm, inv_r, inv_r2, vv, ivjv,
									nv, nvnv, nivnjv, order, inv_c);

				ipnB[j] = pnterms_B(jm, jm2, jm_r, jv2, jv4, -njv, njv2,
									im, im2, im_r, iv2, iv4, -niv, niv2,
									imjm, inv_r, inv_r2, vv, ivjv,
									nv, nvnv, nivnjv, order, inv_c);

				jpnA[j] = pnterms_A(im, im2, im_r, iv2, iv4, +niv, niv2,
									jm, jm2, jm_r, jv2, jv4, +njv, njv2,
									imjm, inv_r, inv_r2, vv, ivjv,
									nv, nvnv, nivnjv, order, inv_c);

				jpnB[j] = pnterms_B(im, im2, im_r, iv2, iv4, +niv, niv2,
									jm, jm2, jm_r, jv2, jv4, +njv, njv2,
									imjm, inv_r, inv_r2, vv, ivjv,
									nv, nvnv, nivnjv, order, inv_c);
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				jp.pnax[j] -= ipnA[j] * rx[j] + ipnB[j] * vx[j];
				jp.pnay[j] -= ipnA[j] * ry[j] + ipnB[j] * vy[j];
				jp.pnaz[j] -= ipnA[j] * rz[j] + ipnB[j] * vz[j];
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				ip.pnax[i] += jpnA[j] * rx[j] + jpnB[j] * vx[j];
				ip.pnay[i] += jpnA[j] * ry[j] + jpnB[j] * vy[j];
				ip.pnaz[i] += jpnA[j] * rz[j] + jpnB[j] * vz[j];
			}
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

				auto inv_r1 = rsqrt(rr);
				inv_r1 = (rr > ee) ? (inv_r1):(0);
				auto inv_r = inv_r1;
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
		real_tn m[LSIZE];
		real_t _m[LSIZE * SIMD];
	};
	union {
		real_tn e2[LSIZE];
		real_t _e2[LSIZE * SIMD];
	};
	union {
		real_tn rx[LSIZE];
		real_t _rx[LSIZE * SIMD];
	};
	union {
		real_tn ry[LSIZE];
		real_t _ry[LSIZE * SIMD];
	};
	union {
		real_tn rz[LSIZE];
		real_t _rz[LSIZE * SIMD];
	};
	union {
		real_tn vx[LSIZE];
		real_t _vx[LSIZE * SIMD];
	};
	union {
		real_tn vy[LSIZE];
		real_t _vy[LSIZE * SIMD];
	};
	union {
		real_tn vz[LSIZE];
		real_t _vz[LSIZE * SIMD];
	};
	union {
		real_tn pnax[LSIZE];
		real_t _pnax[LSIZE * SIMD];
	};
	union {
		real_tn pnay[LSIZE];
		real_t _pnay[LSIZE * SIMD];
	};
	union {
		real_tn pnaz[LSIZE];
		real_t _pnaz[LSIZE * SIMD];
	};
} PNAcc_Data;


static inline void
pnacc_kernel_core(
	uint_t i, uint_t j,
	local PNAcc_Data *ip,
	local PNAcc_Data *jp,
	const CLIGHT clight)
// flop count: 36+???
{
	real_tn ee = ip->e2[i] + jp->_e2[j];
	real_tn rx = ip->rx[i] - jp->_rx[j];
	real_tn ry = ip->ry[i] - jp->_ry[j];
	real_tn rz = ip->rz[i] - jp->_rz[j];
	real_tn vx = ip->vx[i] - jp->_vx[j];
	real_tn vy = ip->vy[i] - jp->_vy[j];
	real_tn vz = ip->vz[i] - jp->_vz[j];

	real_tn rr = ee;
	rr += rx * rx + ry * ry + rz * rz;

	real_tn inv_r1 = rsqrt(rr);
	inv_r1 = (rr > ee) ? (inv_r1):(0);
	real_tn inv_r = inv_r1;
	real_tn inv_r2 = inv_r * inv_r;

	real_tn im = ip->m[i];
	real_tn im2 = im * im;
	real_tn im_r = im * inv_r;
	real_tn iv2 = ip->vx[i] * ip->vx[i]
				+ ip->vy[i] * ip->vy[i]
				+ ip->vz[i] * ip->vz[i];
	real_tn iv4 = iv2 * iv2;
	real_tn niv = rx * ip->vx[i]
				+ ry * ip->vy[i]
				+ rz * ip->vz[i];
	niv *= inv_r;
	real_tn niv2 = niv * niv;

	real_tn jm = jp->_m[j];
	real_tn jm2 = jm * jm;
	real_tn jm_r = jm * inv_r;
	real_tn jv2 = jp->_vx[j] * jp->_vx[j]
				+ jp->_vy[j] * jp->_vy[j]
				+ jp->_vz[j] * jp->_vz[j];
	real_tn jv4 = jv2 * jv2;
	real_tn njv = rx * jp->_vx[j]
				+ ry * jp->_vy[j]
				+ rz * jp->_vz[j];
	njv *= inv_r;
	real_tn njv2 = njv * njv;

	real_tn imjm = im * jm;
	real_tn vv = vx * vx
			   + vy * vy
			   + vz * vz;
	real_tn ivjv = ip->vx[i] * jp->_vx[j]
				 + ip->vy[i] * jp->_vy[j]
				 + ip->vz[i] * jp->_vz[j];
	real_tn nv = rx * vx
			   + ry * vy
			   + rz * vz;
	nv *= inv_r;
	real_tn nvnv = nv * nv;
	real_tn nivnjv = niv * njv;

	uint_t order = clight.order;
	real_t inv_c = clight.inv1;

	real_tn jpnA = pnterms_A(im, im2, im_r, iv2, iv4, +niv, niv2,
							 jm, jm2, jm_r, jv2, jv4, +njv, njv2,
							 imjm, inv_r, inv_r2, vv, ivjv,
							 nv, nvnv, nivnjv, order, inv_c);

	real_tn jpnB = pnterms_B(im, im2, im_r, iv2, iv4, +niv, niv2,
							 jm, jm2, jm_r, jv2, jv4, +njv, njv2,
							 imjm, inv_r, inv_r2, vv, ivjv,
							 nv, nvnv, nivnjv, order, inv_c);

	ip->pnax[i] += jpnA * rx + jpnB * vx;
	ip->pnay[i] += jpnA * ry + jpnB * vy;
	ip->pnaz[i] += jpnA * rz + jpnB * vz;
}


#endif	// __cplusplus
#endif	// __PNACC_KERNEL_COMMON_H__
