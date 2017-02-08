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
			auto im = ip.m[i];
			auto iee = ip.e2[i];
			auto irx = ip.rx[i];
			auto iry = ip.ry[i];
			auto irz = ip.rz[i];
			auto ivx = ip.vx[i];
			auto ivy = ip.vy[i];
			auto ivz = ip.vz[i];
			auto ipnax = ip.pnax[i];
			auto ipnay = ip.pnay[i];
			auto ipnaz = ip.pnaz[i];
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto ee = iee + jp.e2[j];
				auto rx = irx - jp.rx[j];
				auto ry = iry - jp.ry[j];
				auto rz = irz - jp.rz[j];
				auto vx = ivx - jp.vx[j];
				auto vy = ivy - jp.vy[j];
				auto vz = ivz - jp.vz[j];

				auto rr = ee;
				rr += rx * rx + ry * ry + rz * rz;

				auto inv_r = rsqrt(rr);
				auto inv_r2 = inv_r * inv_r;

//				auto im = ip.m[i];
				auto im2 = im * im;
				auto im_r = im * inv_r;
				auto iv2 = ivx * ivx
						 + ivy * ivy
						 + ivz * ivz;
				auto iv4 = iv2 * iv2;
				auto niv = rx * ivx
						 + ry * ivy
						 + rz * ivz;
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
				auto ivjv = ivx * jp.vx[j]
						  + ivy * jp.vy[j]
						  + ivz * jp.vz[j];
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


#endif	// __cplusplus
#endif	// __PNACC_KERNEL_COMMON_H__
