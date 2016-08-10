#ifndef __PNACC_KERNEL_COMMON_H__
#define __PNACC_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"
#include "pn_terms.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
template<typename I, typename J, typename PARAM>
static inline void
p2p_pnacc_kernel_core(I &ip, J &jp, const PARAM clight)
// flop count: 48 + ???
{
	auto rx = ip.rx - jp.rx;
	auto ry = ip.ry - jp.ry;
	auto rz = ip.rz - jp.rz;
	auto vx = ip.vx - jp.vx;
	auto vy = ip.vy - jp.vy;
	auto vz = ip.vz - jp.vz;
	auto e2 = ip.e2 + jp.e2;
	auto r2 = rx * rx + ry * ry + rz * rz;
	auto v2 = vx * vx + vy * vy + vz * vz;

	decltype(r2) inv_r1;
	auto inv_r2 = smoothed_inv_r2_inv_r1(r2, e2, &inv_r1);	// flop count: 4

	auto nx = rx * inv_r1;
	auto ny = ry * inv_r1;
	auto nz = rz * inv_r1;

	{	// i-particle
		auto pn = p2p_pnterms(
			ip.m, ip.vx, ip.vy, ip.vz,
			jp.m, jp.vx, jp.vy, jp.vz,
			nx, ny, nz, vx, vy, vz,
			v2, inv_r1, inv_r2, clight
		);	// flop count: ???

		ip.pnax += (pn.a * nx + pn.b * vx);
		ip.pnay += (pn.a * ny + pn.b * vy);
		ip.pnaz += (pn.a * nz + pn.b * vz);
	}
	{	// j-particle
		auto pn = p2p_pnterms(
			jp.m, jp.vx, jp.vy, jp.vz,
			ip.m, ip.vx, ip.vy, ip.vz,
			-nx, -ny, -nz, -vx, -vy, -vz,
			v2, inv_r1, inv_r2, clight
		);	// flop count: ???

		jp.pnax -= (pn.a * nx + pn.b * vx);
		jp.pnay -= (pn.a * ny + pn.b * vy);
		jp.pnaz -= (pn.a * nz + pn.b * vz);
	}
}
#endif

// ----------------------------------------------------------------------------

#define PNACC_IMPLEMENT_STRUCT(N)							\
	typedef struct concat(pnacc_data, N) {					\
		concat(real_t, N) pnax, pnay, pnaz;					\
		concat(real_t, N) rx, ry, rz, vx, vy, vz, e2, m;	\
	} concat(PNAcc_Data, N);

PNACC_IMPLEMENT_STRUCT(1)
#if SIMD > 1
PNACC_IMPLEMENT_STRUCT(SIMD)
#endif
typedef PNAcc_Data1 PNAcc_Data;


static inline vec(PNAcc_Data)
pnacc_kernel_core(vec(PNAcc_Data) ip, PNAcc_Data jp, const CLIGHT clight)
// flop count: 36+???
{
	vec(real_t) rx = ip.rx - jp.rx;
	vec(real_t) ry = ip.ry - jp.ry;
	vec(real_t) rz = ip.rz - jp.rz;
	vec(real_t) vx = ip.vx - jp.vx;
	vec(real_t) vy = ip.vy - jp.vy;
	vec(real_t) vz = ip.vz - jp.vz;
	vec(real_t) e2 = ip.e2 + jp.e2;
	vec(real_t) r2 = rx * rx + ry * ry + rz * rz;
	vec(real_t) v2 = vx * vx + vy * vy + vz * vz;

	vec(real_t) inv_r1;
	vec(real_t) inv_r2 = smoothed_inv_r2_inv_r1(r2, e2, &inv_r1);	// flop count: 4
	inv_r1 = select((vec(real_t))(0), inv_r1, (vec(int_t))(r2 > 0));
	inv_r2 = select((vec(real_t))(0), inv_r2, (vec(int_t))(r2 > 0));

	vec(real_t) nx = rx * inv_r1;
	vec(real_t) ny = ry * inv_r1;
	vec(real_t) nz = rz * inv_r1;

	PN pn = p2p_pnterms(
		ip.m, ip.vx, ip.vy, ip.vz,
		jp.m, jp.vx, jp.vy, jp.vz,
		nx, ny, nz, vx, vy, vz,
		v2, inv_r1, inv_r2, clight
	);	// flop count: ???

	ip.pnax += pn.a * nx + pn.b * vx;
	ip.pnay += pn.a * ny + pn.b * vy;
	ip.pnaz += pn.a * nz + pn.b * vz;
	return ip;
}


#endif	// __PNACC_KERNEL_COMMON_H__
