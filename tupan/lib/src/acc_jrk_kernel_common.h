#ifndef __ACC_JRK_KERNEL_COMMON_H__
#define __ACC_JRK_KERNEL_COMMON_H__


#include "common.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
namespace Acc_Jrk {

template<size_t TILE>
struct Acc_Jrk_Data_SoA {
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
};

template<size_t TILE>
auto setup(
	const uint_t n,
	const real_t __m[],
	const real_t __e2[],
	const real_t __rdot[])
{
	auto ntiles = (n + TILE - 1) / TILE;
	vector<Acc_Jrk_Data_SoA<TILE>> part(ntiles);
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
void commit(const uint_t n, const PART& part, real_t __adot[])
{
	for (size_t k = 0; k < n; ++k) {
		auto kk = k%TILE;
		auto& p = part[k/TILE];
		__adot[(0*NDIM+0)*n + k] = p.ax[kk];
		__adot[(0*NDIM+1)*n + k] = p.ay[kk];
		__adot[(0*NDIM+2)*n + k] = p.az[kk];
		__adot[(1*NDIM+0)*n + k] = p.jx[kk];
		__adot[(1*NDIM+1)*n + k] = p.jy[kk];
		__adot[(1*NDIM+2)*n + k] = p.jz[kk];
	}
}

template<size_t TILE>
struct P2P_acc_jrk_kernel_core {
	template<typename IP, typename JP>
	void operator()(IP&& ip, JP&& jp) {
		// flop count: 56
		for (size_t i = 0; i < TILE; ++i) {
			auto im = ip.m[i];
			auto iee = ip.e2[i];
			auto irx = ip.rx[i];
			auto iry = ip.ry[i];
			auto irz = ip.rz[i];
			auto ivx = ip.vx[i];
			auto ivy = ip.vy[i];
			auto ivz = ip.vz[i];
			auto iax = ip.ax[i];
			auto iay = ip.ay[i];
			auto iaz = ip.az[i];
			auto ijx = ip.jx[i];
			auto ijy = ip.jy[i];
			auto ijz = ip.jz[i];
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

				auto inv_r3 = rsqrt(rr);
				auto inv_r2 = inv_r3 * inv_r3;
				inv_r3 *= inv_r2;
				inv_r2 *= -3;

				auto s1 = rx * vx + ry * vy + rz * vz;

				auto q1 = inv_r2 * (s1);
				vx += q1 * rx;
				vy += q1 * ry;
				vz += q1 * rz;

				auto im_r3 = im * inv_r3;
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


typedef struct acc_jrk_data {
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
		real_tn ax[LSIZE];
		real_t _ax[LSIZE * SIMD];
	};
	union {
		real_tn ay[LSIZE];
		real_t _ay[LSIZE * SIMD];
	};
	union {
		real_tn az[LSIZE];
		real_t _az[LSIZE * SIMD];
	};
	union {
		real_tn jx[LSIZE];
		real_t _jx[LSIZE * SIMD];
	};
	union {
		real_tn jy[LSIZE];
		real_t _jy[LSIZE * SIMD];
	};
	union {
		real_tn jz[LSIZE];
		real_t _jz[LSIZE * SIMD];
	};
} Acc_Jrk_Data;


#endif	// __cplusplus
#endif	// __ACC_JRK_KERNEL_COMMON_H__
