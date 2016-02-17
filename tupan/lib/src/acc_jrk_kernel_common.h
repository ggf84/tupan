#ifndef __ACC_JRK_KERNEL_COMMON_H__
#define __ACC_JRK_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
static inline void
p2p_acc_jrk_kernel_core(auto &ip, auto &jp)
// flop count: 56
{
	auto rx = ip.rx - jp.rx;
	auto ry = ip.ry - jp.ry;
	auto rz = ip.rz - jp.rz;
	auto vx = ip.vx - jp.vx;
	auto vy = ip.vy - jp.vy;
	auto vz = ip.vz - jp.vz;
	auto e2 = ip.e2 + jp.e2;
	auto r2 = rx * rx + ry * ry + rz * rz;
	auto rv = rx * vx + ry * vy + rz * vz;

	decltype(r2) inv_r2;
	auto inv_r3 = smoothed_inv_r3_inv_r2(r2, e2, &inv_r2);	// flop count: 5

	auto alpha = 3 * rv * inv_r2;

	vx -= alpha * rx;
	vy -= alpha * ry;
	vz -= alpha * rz;

	{	// i-particle
		auto m_r3 = jp.m * inv_r3;
		ip.ax -= m_r3 * rx;
		ip.ay -= m_r3 * ry;
		ip.az -= m_r3 * rz;
		ip.jx -= m_r3 * vx;
		ip.jy -= m_r3 * vy;
		ip.jz -= m_r3 * vz;
	}
	{	// j-particle
		auto m_r3 = ip.m * inv_r3;
		jp.ax += m_r3 * rx;
		jp.ay += m_r3 * ry;
		jp.az += m_r3 * rz;
		jp.jx += m_r3 * vx;
		jp.jy += m_r3 * vy;
		jp.jz += m_r3 * vz;
	}
}
#endif

// ----------------------------------------------------------------------------

#define ACC_JRK_IMPLEMENT_STRUCT(N)							\
	typedef struct concat(acc_jrk_data, N) {				\
		concat(real_t, N) ax, ay, az, jx, jy, jz;			\
		concat(real_t, N) rx, ry, rz, vx, vy, vz, e2, m;	\
	} concat(Acc_Jrk_Data, N);

ACC_JRK_IMPLEMENT_STRUCT(1)
#if SIMD > 1
ACC_JRK_IMPLEMENT_STRUCT(SIMD)
#endif
typedef Acc_Jrk_Data1 Acc_Jrk_Data;


static inline vec(Acc_Jrk_Data)
acc_jrk_kernel_core(vec(Acc_Jrk_Data) ip, Acc_Jrk_Data jp)
// flop count: 43
{
	vec(real_t) rx = ip.rx - jp.rx;
	vec(real_t) ry = ip.ry - jp.ry;
	vec(real_t) rz = ip.rz - jp.rz;
	vec(real_t) vx = ip.vx - jp.vx;
	vec(real_t) vy = ip.vy - jp.vy;
	vec(real_t) vz = ip.vz - jp.vz;
	vec(real_t) e2 = ip.e2 + jp.e2;
	vec(real_t) r2 = rx * rx + ry * ry + rz * rz;
	vec(real_t) rv = rx * vx + ry * vy + rz * vz;

	vec(real_t) inv_r2;
	vec(real_t) m_r3 = jp.m * smoothed_inv_r3_inv_r2(r2, e2, &inv_r2);	// flop count: 6
	inv_r2 = select((vec(real_t))(0), inv_r2, (r2 > 0));
	m_r3 = select((vec(real_t))(0), m_r3, (r2 > 0));

	vec(real_t) alpha = 3 * rv * inv_r2;

	vx -= alpha * rx;
	vy -= alpha * ry;
	vz -= alpha * rz;

	ip.ax -= m_r3 * rx;
	ip.ay -= m_r3 * ry;
	ip.az -= m_r3 * rz;
	ip.jx -= m_r3 * vx;
	ip.jy -= m_r3 * vy;
	ip.jz -= m_r3 * vz;
	return ip;
}


#endif	// __ACC_JRK_KERNEL_COMMON_H__
