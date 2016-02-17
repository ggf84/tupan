#ifndef __PHI_KERNEL_COMMON_H__
#define __PHI_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
static inline void
p2p_phi_kernel_core(auto &ip, auto &jp)
// flop count: 16
{
	auto rx = ip.rx - jp.rx;
	auto ry = ip.ry - jp.ry;
	auto rz = ip.rz - jp.rz;
	auto e2 = ip.e2 + jp.e2;
	auto r2 = rx * rx + ry * ry + rz * rz;
	auto inv_r1 = smoothed_inv_r1(r2, e2);	// flop count: 3

	ip.phi -= jp.m * inv_r1;
	jp.phi -= ip.m * inv_r1;
}
#endif

// ----------------------------------------------------------------------------

#define PHI_IMPLEMENT_STRUCT(N)					\
	typedef struct concat(phi_data, N) {		\
		concat(real_t, N) phi;					\
		concat(real_t, N) rx, ry, rz, e2, m;	\
	} concat(Phi_Data, N);

PHI_IMPLEMENT_STRUCT(1)
#if SIMD > 1
PHI_IMPLEMENT_STRUCT(SIMD)
#endif
typedef Phi_Data1 Phi_Data;


static inline vec(Phi_Data)
phi_kernel_core(vec(Phi_Data) ip, Phi_Data jp)
// flop count: 14
{
	vec(real_t) rx = ip.rx - jp.rx;
	vec(real_t) ry = ip.ry - jp.ry;
	vec(real_t) rz = ip.rz - jp.rz;
	vec(real_t) e2 = ip.e2 + jp.e2;
	vec(real_t) r2 = rx * rx + ry * ry + rz * rz;

	vec(real_t) inv_r1 = smoothed_inv_r1(r2, e2);	// flop count: 3
	inv_r1 = select((vec(real_t))(0), inv_r1, (r2 > 0));

	ip.phi -= jp.m * inv_r1;
	return ip;
}


#endif	// __PHI_KERNEL_COMMON_H__
