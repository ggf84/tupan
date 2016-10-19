#ifndef __PNACC_KERNEL_COMMON_H__
#define __PNACC_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"
#include "pn_terms.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
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

template<size_t TILE, typename T = PNAcc_Data_SoA<TILE>>
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
				auto rr = ip.e2[i] + jp.e2[j];
				rx[j] = ip.rx[i] - jp.rx[j];
				ry[j] = ip.ry[i] - jp.ry[j];
				rz[j] = ip.rz[i] - jp.rz[j];
				vx[j] = ip.vx[i] - jp.vx[j];
				vy[j] = ip.vy[i] - jp.vy[j];
				vz[j] = ip.vz[i] - jp.vz[j];

				rr += rx[j] * rx[j] + ry[j] * ry[j] + rz[j] * rz[j];

				inv_r1[j] = rsqrt(rr);
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto inv_r2 = inv_r1[j] * inv_r1[j];

				auto vv = vx[j] * vx[j] + vy[j] * vy[j] + vz[j] * vz[j];

				ipnA[j] = pnterms_A(
					jp.m[j], jp.vx[j], jp.vy[j], jp.vz[j],
					ip.m[i], ip.vx[i], ip.vy[i], ip.vz[i],
					-rx[j], -ry[j], -rz[j], -vx[j], -vy[j], -vz[j],
					vv, inv_r1[j], inv_r2, clight.order, clight.inv1
				);	// flop count: ???

				ipnB[j] = pnterms_B(
					jp.m[j], jp.vx[j], jp.vy[j], jp.vz[j],
					ip.m[i], ip.vx[i], ip.vy[i], ip.vz[i],
					-rx[j], -ry[j], -rz[j], -vx[j], -vy[j], -vz[j],
					vv, inv_r1[j], inv_r2, clight.order, clight.inv1
				);	// flop count: ???

				jpnA[j] = pnterms_A(
					ip.m[i], ip.vx[i], ip.vy[i], ip.vz[i],
					jp.m[j], jp.vx[j], jp.vy[j], jp.vz[j],
					rx[j], ry[j], rz[j], vx[j], vy[j], vz[j],
					vv, inv_r1[j], inv_r2, clight.order, clight.inv1
				);	// flop count: ???

				jpnB[j] = pnterms_B(
					ip.m[i], ip.vx[i], ip.vy[i], ip.vz[i],
					jp.m[j], jp.vx[j], jp.vy[j], jp.vz[j],
					rx[j], ry[j], rz[j], vx[j], vy[j], vz[j],
					vv, inv_r1[j], inv_r2, clight.order, clight.inv1
				);	// flop count: ???
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
		decltype(p.m) rx, ry, rz, vx, vy, vz, inv_r1;
		decltype(p.m) ipnA, ipnB;
		for (size_t i = 0; i < TILE; ++i) {
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto rr = p.e2[i] + p.e2[j];
				rx[j] = p.rx[i] - p.rx[j];
				ry[j] = p.ry[i] - p.ry[j];
				rz[j] = p.rz[i] - p.rz[j];
				vx[j] = p.vx[i] - p.vx[j];
				vy[j] = p.vy[i] - p.vy[j];
				vz[j] = p.vz[i] - p.vz[j];

				rr += rx[j] * rx[j] + ry[j] * ry[j] + rz[j] * rz[j];

				inv_r1[j] = rsqrt(rr);
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				inv_r1[j] = (i == j) ? (0):(inv_r1[j]);
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				auto inv_r2 = inv_r1[j] * inv_r1[j];

				auto vv = vx[j] * vx[j] + vy[j] * vy[j] + vz[j] * vz[j];

				ipnA[j] = pnterms_A(
					p.m[j], p.vx[j], p.vy[j], p.vz[j],
					p.m[i], p.vx[i], p.vy[i], p.vz[i],
					-rx[j], -ry[j], -rz[j], -vx[j], -vy[j], -vz[j],
					vv, inv_r1[j], inv_r2, clight.order, clight.inv1
				);	// flop count: ???

				ipnB[j] = pnterms_B(
					p.m[j], p.vx[j], p.vy[j], p.vz[j],
					p.m[i], p.vx[i], p.vy[i], p.vz[i],
					-rx[j], -ry[j], -rz[j], -vx[j], -vy[j], -vz[j],
					vv, inv_r1[j], inv_r2, clight.order, clight.inv1
				);	// flop count: ???
			}
			#pragma omp simd
			for (size_t j = 0; j < TILE; ++j) {
				p.pnax[j] -= ipnA[j] * rx[j] + ipnB[j] * vx[j];
				p.pnay[j] -= ipnA[j] * ry[j] + ipnB[j] * vy[j];
				p.pnaz[j] -= ipnA[j] * rz[j] + ipnB[j] * vz[j];
			}
		}
	}
};
#endif

// ----------------------------------------------------------------------------

#define PNACC_IMPLEMENT_STRUCT(N)			\
	typedef struct concat(pnacc_data, N) {	\
		concat(real_t, N) m, e2;			\
		concat(real_t, N) rdot[2][NDIM];	\
		concat(real_t, N) pnacc[NDIM];		\
	} concat(PNAcc_Data, N);

PNACC_IMPLEMENT_STRUCT(1)
#if SIMD > 1
PNACC_IMPLEMENT_STRUCT(SIMD)
#endif
typedef PNAcc_Data1 PNAcc_Data;


static inline vec(PNAcc_Data)
pnacc_kernel_core(vec(PNAcc_Data) ip, vec(PNAcc_Data) jp, const CLIGHT clight)
// flop count: 36+???
{
	vec(real_t) rdot[2][NDIM];
	#pragma unroll
	for (uint_t kdot = 0; kdot < 2; ++kdot) {
		#pragma unroll
		for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
			rdot[kdot][kdim] = ip.rdot[kdot][kdim] - jp.rdot[kdot][kdim];
		}
	}
	vec(real_t) e2 = ip.e2 + jp.e2;

	vec(real_t) rr = rdot[0][0] * rdot[0][0]
			+ rdot[0][1] * rdot[0][1]
			+ rdot[0][2] * rdot[0][2];
	vec(real_t) vv = rdot[1][0] * rdot[1][0]
			+ rdot[1][1] * rdot[1][1]
			+ rdot[1][2] * rdot[1][2];

	vec(real_t) inv_r1;
	vec(real_t) inv_r2 = smoothed_inv_r2_inv_r1(rr, e2, &inv_r1);	// flop count: 4
	inv_r1 = select((vec(real_t))(0), inv_r1, (vec(int_t))(rr > 0));
	inv_r2 = select((vec(real_t))(0), inv_r2, (vec(int_t))(rr > 0));

	vec(real_t) pnA = pnterms_A(
		ip.m, ip.rdot[1][0], ip.rdot[1][1], ip.rdot[1][2],
		jp.m, jp.rdot[1][0], jp.rdot[1][1], jp.rdot[1][2],
		rdot[0][0], rdot[0][1], rdot[0][2],
		rdot[1][0], rdot[1][1], rdot[1][2],
		vv, inv_r1, inv_r2, clight.order, clight.inv1
	);	// flop count: ???

	vec(real_t) pnB = pnterms_B(
		ip.m, ip.rdot[1][0], ip.rdot[1][1], ip.rdot[1][2],
		jp.m, jp.rdot[1][0], jp.rdot[1][1], jp.rdot[1][2],
		rdot[0][0], rdot[0][1], rdot[0][2],
		rdot[1][0], rdot[1][1], rdot[1][2],
		vv, inv_r1, inv_r2, clight.order, clight.inv1
	);	// flop count: ???

	#pragma unroll
	for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
		ip.pnacc[kdim] += (pnA * rdot[0][kdim] + pnB * rdot[1][kdim]);
	}
	return ip;
}


#endif	// __PNACC_KERNEL_COMMON_H__
