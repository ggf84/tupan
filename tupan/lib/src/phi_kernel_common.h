#ifndef __PHI_KERNEL_COMMON_H__
#define __PHI_KERNEL_COMMON_H__


#include "common.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
namespace Phi {

template<size_t TILE>
struct Phi_Data_SoA {
	real_t __ALIGNED__ m[TILE];
	real_t __ALIGNED__ e2[TILE];
	real_t __ALIGNED__ rx[TILE];
	real_t __ALIGNED__ ry[TILE];
	real_t __ALIGNED__ rz[TILE];
	real_t __ALIGNED__ phi[TILE];
};

template<size_t TILE>
auto setup(
	const uint_t n,
	const real_t __m[],
	const real_t __e2[],
	const real_t __rdot[])
{
	auto ntiles = (n + TILE - 1) / TILE;
	vector<Phi_Data_SoA<TILE>> part(ntiles);
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
void commit(const uint_t n, const PART& part, real_t __phi[])
{
	for (size_t k = 0; k < n; ++k) {
		auto kk = k%TILE;
		auto& p = part[k/TILE];
		__phi[k] = p.phi[kk];
	}
}

template<size_t TILE>
struct P2P_phi_kernel_core {
	template<typename IP, typename JP>
	void operator()(IP&& ip, JP&& jp) {
		// flop count: 16
		for (size_t i = 0; i < TILE; ++i) {
			auto im = ip.m[i];
			auto iee = ip.e2[i];
			auto irx = ip.rx[i];
			auto iry = ip.ry[i];
			auto irz = ip.rz[i];
			auto iphi = ip.phi[i];
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto ee = iee + jp.e2[j];
				auto rx = irx - jp.rx[j];
				auto ry = iry - jp.ry[j];
				auto rz = irz - jp.rz[j];

				auto rr = ee;
				rr += rx * rx + ry * ry + rz * rz;

				auto inv_r1 = rsqrt(rr);

				jp.phi[j] -= im * inv_r1;

				iphi -= jp.m[j] * inv_r1;
			}
			ip.phi[i] = iphi;
		}
	}

	template<typename P>
	void operator()(P&& p) {
		// flop count: 14
		for (size_t i = 0; i < TILE; ++i) {
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto ee = p.e2[i] + p.e2[j];
				auto rx = p.rx[i] - p.rx[j];
				auto ry = p.ry[i] - p.ry[j];
				auto rz = p.rz[i] - p.rz[j];

				auto rr = ee;
				rr += rx * rx + ry * ry + rz * rz;

				auto inv_r1 = rsqrt(rr);
				inv_r1 = (rr > ee) ? (inv_r1):(0);

				p.phi[j] -= p.m[i] * inv_r1;
			}
		}
	}
};

}	// namespace Phi
#else


// ----------------------------------------------------------------------------


typedef struct phi_data {
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
		real_tn phi[LSIZE];
		real_t _phi[LSIZE * SIMD];
	};
} Phi_Data;


#endif	// __cplusplus
#endif	// __PHI_KERNEL_COMMON_H__
