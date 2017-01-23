#ifndef __ACC_KERNEL_COMMON_H__
#define __ACC_KERNEL_COMMON_H__


#include "common.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
namespace Acc {

template<size_t TILE>
struct Acc_Data_SoA {
	real_t m[TILE];
	real_t e2[TILE];
	real_t rx[TILE];
	real_t ry[TILE];
	real_t rz[TILE];
	real_t ax[TILE];
	real_t ay[TILE];
	real_t az[TILE];
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
		__adot[(0*NDIM+0)*n + k] = p.ax[kk];
		__adot[(0*NDIM+1)*n + k] = p.ay[kk];
		__adot[(0*NDIM+2)*n + k] = p.az[kk];
	}
}

template<size_t TILE>
struct P2P_acc_kernel_core {
	template<typename IP, typename JP>
	void operator()(IP&& ip, JP&& jp) {
		// flop count: 28
		decltype(jp.m) rx, ry, rz, inv_r3, im_r3, jm_r3;
		for (size_t i = 0; i < TILE; ++i) {
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto ee = ip.e2[i] + jp.e2[j];
				rx[j] = ip.rx[i] - jp.rx[j];
				ry[j] = ip.ry[i] - jp.ry[j];
				rz[j] = ip.rz[i] - jp.rz[j];

				auto rr = ee;
				rr += rx[j] * rx[j] + ry[j] * ry[j] + rz[j] * rz[j];

				inv_r3[j] = rsqrt(rr);
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				inv_r3[j] *= inv_r3[j] * inv_r3[j];

				im_r3[j] = ip.m[i] * inv_r3[j];
				jm_r3[j] = jp.m[j] * inv_r3[j];
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				jp.ax[j] += im_r3[j] * rx[j];
				jp.ay[j] += im_r3[j] * ry[j];
				jp.az[j] += im_r3[j] * rz[j];
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				ip.ax[i] -= jm_r3[j] * rx[j];
				ip.ay[i] -= jm_r3[j] * ry[j];
				ip.az[i] -= jm_r3[j] * rz[j];
			}
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


typedef struct acc_data {
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
} Acc_Data;


#endif	// __cplusplus
#endif	// __ACC_KERNEL_COMMON_H__
