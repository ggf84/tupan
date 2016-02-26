#ifndef __TSTEP_KERNEL_COMMON_H__
#define __TSTEP_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
static inline void
p2p_tstep_kernel_core(auto &ip, auto &jp, const auto eta)
// flop count: 43
{
	auto rx = ip.rx - jp.rx;
	auto ry = ip.ry - jp.ry;
	auto rz = ip.rz - jp.rz;
	auto vx = ip.vx - jp.vx;
	auto vy = ip.vy - jp.vy;
	auto vz = ip.vz - jp.vz;
	auto e2 = ip.e2 + jp.e2;
	auto m = ip.m + jp.m;
	auto r2 = rx * rx + ry * ry + rz * rz;
	auto rv = rx * vx + ry * vy + rz * vz;
	auto v2 = vx * vx + vy * vy + vz * vz;

	decltype(r2) inv_r2;
	auto m_r3 = 2 * m * smoothed_inv_r3_inv_r2(r2, e2, &inv_r2);	// flop count: 7

	auto m_r5 = m_r3 * inv_r2;
	auto w2 = m_r3 + v2 * inv_r2;
	auto gamma = m_r5 + w2 * inv_r2;
	gamma *= (eta * rsqrt(w2));
	w2 -= gamma * rv;

	ip.w2_a += w2;
	jp.w2_a += w2;
	ip.w2_b = fmax(w2, ip.w2_b);
	jp.w2_b = fmax(w2, jp.w2_b);
}
#endif

// ----------------------------------------------------------------------------

#define TSTEP_IMPLEMENT_STRUCT(N)							\
	typedef struct concat(tstep_data, N) {					\
		concat(real_t, N) w2_a, w2_b;						\
		concat(real_t, N) rx, ry, rz, vx, vy, vz, e2, m;	\
	} concat(Tstep_Data, N);

TSTEP_IMPLEMENT_STRUCT(1)
#if SIMD > 1
TSTEP_IMPLEMENT_STRUCT(SIMD)
#endif
typedef Tstep_Data1 Tstep_Data;


static inline vec(Tstep_Data)
tstep_kernel_core(vec(Tstep_Data) ip, Tstep_Data jp, const real_t eta)
// flop count: 42
{
	vec(real_t) rx = ip.rx - jp.rx;
	vec(real_t) ry = ip.ry - jp.ry;
	vec(real_t) rz = ip.rz - jp.rz;
	vec(real_t) vx = ip.vx - jp.vx;
	vec(real_t) vy = ip.vy - jp.vy;
	vec(real_t) vz = ip.vz - jp.vz;
	vec(real_t) e2 = ip.e2 + jp.e2;
	vec(real_t) m = ip.m + jp.m;
	vec(real_t) r2 = rx * rx + ry * ry + rz * rz;
	vec(real_t) rv = rx * vx + ry * vy + rz * vz;
	vec(real_t) v2 = vx * vx + vy * vy + vz * vz;

	vec(real_t) inv_r2;
	vec(real_t) m_r3 = 2 * m * smoothed_inv_r3_inv_r2(r2, e2, &inv_r2);	// flop count: 7

	vec(real_t) m_r5 = m_r3 * inv_r2;
	vec(real_t) w2 = m_r3 + v2 * inv_r2;
	vec(real_t) gamma = m_r5 + w2 * inv_r2;
	gamma *= (eta * rsqrt(w2));
	w2 -= gamma * rv;

	w2 = select((vec(real_t))(0), w2, (vec(int_t))(r2 > 0));

	ip.w2_a += w2;
	ip.w2_b = fmax(w2, ip.w2_b);
	return ip;
}


#endif	// __TSTEP_KERNEL_COMMON_H__
