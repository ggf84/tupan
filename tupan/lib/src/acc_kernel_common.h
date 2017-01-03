#ifndef __ACC_KERNEL_COMMON_H__
#define __ACC_KERNEL_COMMON_H__


#include "common.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
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

template<size_t TILE, typename T = Acc_Data_SoA<TILE>>
auto setup(
	const uint_t n,
	const real_t __m[],
	const real_t __e2[],
	const real_t __rdot[])
{
	auto ntiles = (n + TILE - 1) / TILE;
	vector<T> part(ntiles);
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
#endif


// ----------------------------------------------------------------------------


typedef struct acc_data {
	union {
		real_t _m[SIMD];
		real_tn m;
	};
	union {
		real_t _e2[SIMD];
		real_tn e2;
	};
	union {
		real_t _rx[SIMD];
		real_tn rx;
	};
	union {
		real_t _ry[SIMD];
		real_tn ry;
	};
	union {
		real_t _rz[SIMD];
		real_tn rz;
	};
	union {
		real_t _ax[SIMD];
		real_tn ax;
	};
	union {
		real_t _ay[SIMD];
		real_tn ay;
	};
	union {
		real_t _az[SIMD];
		real_tn az;
	};
} Acc_Data;


static inline Acc_Data
acc_kernel_core(Acc_Data ip, Acc_Data jp)
// flop count: 21
{
	real_tn ee = ip.e2 + jp.e2;
	real_tn rx = ip.rx - jp.rx;
	real_tn ry = ip.ry - jp.ry;
	real_tn rz = ip.rz - jp.rz;

	real_tn rr = ee;
	rr += rx * rx + ry * ry + rz * rz;

	real_tn inv_r3 = rsqrt(rr);
	inv_r3 = (rr > ee) ? (inv_r3):(0);
	inv_r3 *= inv_r3 * inv_r3;

	real_tn jm_r3 = jp.m * inv_r3;

	ip.ax -= jm_r3 * rx;
	ip.ay -= jm_r3 * ry;
	ip.az -= jm_r3 * rz;
	return ip;
}


#endif	// __ACC_KERNEL_COMMON_H__
