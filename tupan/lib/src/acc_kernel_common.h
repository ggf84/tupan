#ifndef __ACC_KERNEL_COMMON_H__
#define __ACC_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#define ACC_DECL_STRUCTS(T, iT, jT)	\
	typedef struct acc_data {		\
		T ax, ay, az;				\
		T rx, ry, rz, e2, m;		\
	} Acc_Data;						\
	typedef struct acc_idata {		\
		iT ax, ay, az;				\
		iT rx, ry, rz, e2, m;		\
	} Acc_IData;					\
	typedef struct acc_jdata {		\
		jT rx, ry, rz, e2, m;		\
	} Acc_JData;

ACC_DECL_STRUCTS(real_tn, real_tn, real_t)


#ifdef __cplusplus	// not for OpenCL
struct p2p_acc_kernel_core {
	// flop count: 28
	void operator()(auto &i, auto &j) {
		auto rx = i.rx - j.rx;
		auto ry = i.ry - j.ry;
		auto rz = i.rz - j.rz;
		auto e2 = i.e2 + j.e2;
		auto r2 = rx * rx + ry * ry + rz * rz;
		auto inv_r3 = smoothed_inv_r3(r2, e2);	// flop count: 5
		{	// i-particle
			auto m_r3 = j.m * inv_r3;
			i.ax -= m_r3 * rx;
			i.ay -= m_r3 * ry;
			i.az -= m_r3 * rz;
		}
		{	// j-particle
			auto m_r3 = i.m * inv_r3;
			j.ax += m_r3 * rx;
			j.ay += m_r3 * ry;
			j.az += m_r3 * rz;
		}
	}
};
#endif


static inline Acc_IData
acc_kernel_core(Acc_IData ip, Acc_JData jp)
// flop count: 21
{
	real_tn rx = ip.rx - jp.rx;
	real_tn ry = ip.ry - jp.ry;
	real_tn rz = ip.rz - jp.rz;
	real_tn e2 = ip.e2 + jp.e2;
	real_tn r2 = rx * rx + ry * ry + rz * rz;
	int_tn mask = (r2 > 0);

	real_tn m_r3 = smoothed_m_r3(jp.m, r2, e2, mask);	// flop count: 6

	ip.ax -= m_r3 * rx;
	ip.ay -= m_r3 * ry;
	ip.az -= m_r3 * rz;
	return ip;
}

#endif	// __ACC_KERNEL_COMMON_H__
