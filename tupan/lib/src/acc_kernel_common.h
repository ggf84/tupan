#ifndef __ACC_KERNEL_COMMON_H__
#define __ACC_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#ifdef __cplusplus	// cpp only, i.e., not for OpenCL
static inline void
p2p_acc_kernel_core(auto &ip, auto &jp)
// flop count: 28
{
	auto rx = ip.rx - jp.rx;
	auto ry = ip.ry - jp.ry;
	auto rz = ip.rz - jp.rz;
	auto e2 = ip.e2 + jp.e2;
	auto r2 = rx * rx + ry * ry + rz * rz;

	auto inv_r3 = smoothed_inv_r3(r2, e2);	// flop count: 5

	{	// i-particle
		auto m_r3 = jp.m * inv_r3;
		ip.ax -= m_r3 * rx;
		ip.ay -= m_r3 * ry;
		ip.az -= m_r3 * rz;
	}
	{	// j-particle
		auto m_r3 = ip.m * inv_r3;
		jp.ax += m_r3 * rx;
		jp.ay += m_r3 * ry;
		jp.az += m_r3 * rz;
	}
}
#endif

// ----------------------------------------------------------------------------

#define ACC_IMPLEMENT_STRUCT(N)					\
	typedef struct concat(acc_data, N) {		\
		concat(real_t, N) ax, ay, az;			\
		concat(real_t, N) rx, ry, rz, e2, m;	\
	} concat(Acc_Data, N);

ACC_IMPLEMENT_STRUCT(1)
#if SIMD > 1
ACC_IMPLEMENT_STRUCT(SIMD)
#endif
typedef Acc_Data1 Acc_Data;


static inline vec(Acc_Data)
acc_kernel_core(vec(Acc_Data) ip, Acc_Data jp)
// flop count: 21
{
	vec(real_t) rx = ip.rx - jp.rx;
	vec(real_t) ry = ip.ry - jp.ry;
	vec(real_t) rz = ip.rz - jp.rz;
	vec(real_t) e2 = ip.e2 + jp.e2;
	vec(real_t) r2 = rx * rx + ry * ry + rz * rz;

	vec(real_t) m_r3 = jp.m * smoothed_inv_r3(r2, e2);	// flop count: 6
	m_r3 = select((vec(real_t))(0), m_r3, (vec(int_t))(r2 > 0));

	ip.ax -= m_r3 * rx;
	ip.ay -= m_r3 * ry;
	ip.az -= m_r3 * rz;
	return ip;
}


#endif	// __ACC_KERNEL_COMMON_H__
