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
		for (size_t i = 0; i < TILE; ++i) {
			#pragma unroll
			for (size_t j = 0; j < TILE; ++j) {
				auto rr = ip.e2[i] + jp.e2[j];
				auto rx = ip.rx[i] - jp.rx[j];
				auto ry = ip.ry[i] - jp.ry[j];
				auto rz = ip.rz[i] - jp.rz[j];
				auto vx = ip.vx[i] - jp.vx[j];
				auto vy = ip.vy[i] - jp.vy[j];
				auto vz = ip.vz[i] - jp.vz[j];

				rr += rx * rx + ry * ry + rz * rz;
				auto vv = vx * vx + vy * vy + vz * vz;

				auto inv_r1 = rsqrt(rr);
				auto inv_r2 = inv_r1 * inv_r1;

				// i-particle
				auto jpn = p2p_pnterms(
					ip.m[i], ip.vx[i], ip.vy[i], ip.vz[i],
					jp.m[j], jp.vx[j], jp.vy[j], jp.vz[j],
					rx, ry, rz, vx, vy, vz,
					vv, inv_r1, inv_r2, clight
				);	// flop count: ???
				ip.pnax[i] += jpn.a * rx + jpn.b * vx;
				ip.pnay[i] += jpn.a * ry + jpn.b * vy;
				ip.pnaz[i] += jpn.a * rz + jpn.b * vz;

				// j-particle
				auto ipn = p2p_pnterms(
					jp.m[j], jp.vx[j], jp.vy[j], jp.vz[j],
					ip.m[i], ip.vx[i], ip.vy[i], ip.vz[i],
					-rx, -ry, -rz, -vx, -vy, -vz,
					vv, inv_r1, inv_r2, clight
				);	// flop count: ???
				jp.pnax[j] -= ipn.a * rx + ipn.b * vx;
				jp.pnay[j] -= ipn.a * ry + ipn.b * vy;
				jp.pnaz[j] -= ipn.a * rz + ipn.b * vz;
			}
		}
	}

	template<typename P>
	void operator()(P&& p) {
		// flop count: 33 + ???
		for (size_t i = 0; i < TILE; ++i) {
			#pragma unroll
			for (size_t j = 0; j < TILE; ++j) {
				auto rr = p.e2[i] + p.e2[j];
				auto rx = p.rx[i] - p.rx[j];
				auto ry = p.ry[i] - p.ry[j];
				auto rz = p.rz[i] - p.rz[j];
				auto vx = p.vx[i] - p.vx[j];
				auto vy = p.vy[i] - p.vy[j];
				auto vz = p.vz[i] - p.vz[j];

				rr += rx * rx + ry * ry + rz * rz;
				auto vv = vx * vx + vy * vy + vz * vz;

				auto inv_r1 = rsqrt(rr);
				inv_r1 = (i != j) ? (inv_r1):(0);
				auto inv_r2 = inv_r1 * inv_r1;

				auto jpn = p2p_pnterms(
					p.m[i], p.vx[i], p.vy[i], p.vz[i],
					p.m[j], p.vx[j], p.vy[j], p.vz[j],
					rx, ry, rz, vx, vy, vz,
					vv, inv_r1, inv_r2, clight
				);	// flop count: ???
				p.pnax[i] += jpn.a * rx + jpn.b * vx;
				p.pnay[i] += jpn.a * ry + jpn.b * vy;
				p.pnaz[i] += jpn.a * rz + jpn.b * vz;
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

	PN pn = p2p_pnterms(
		ip.m, ip.rdot[1][0], ip.rdot[1][1], ip.rdot[1][2],
		jp.m, jp.rdot[1][0], jp.rdot[1][1], jp.rdot[1][2],
		rdot[0][0], rdot[0][1], rdot[0][2],
		rdot[1][0], rdot[1][1], rdot[1][2],
		vv, inv_r1, inv_r2, clight
	);	// flop count: ???

	#pragma unroll
	for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
		ip.pnacc[kdim] += (pn.a * rdot[0][kdim] + pn.b * rdot[1][kdim]);
	}
	return ip;
}


#endif	// __PNACC_KERNEL_COMMON_H__
