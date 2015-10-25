#ifndef __ACC_KERNEL_COMMON_H__
#define __ACC_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#define ACC_DECL_STRUCTS(iT, jT)	\
	typedef struct acc_idata {		\
		iT ax, ay, az;				\
		iT rx, ry, rz, e2, m;		\
	} Acc_IData;					\
	typedef struct acc_jdata {		\
		jT rx, ry, rz, e2, m;		\
	} Acc_JData;

ACC_DECL_STRUCTS(real_tn, real_t)


static inline Acc_IData
acc_kernel_core(Acc_IData ip, Acc_JData jp)
{
	real_tn rx = ip.rx - jp.rx;													// 1 FLOPs
	real_tn ry = ip.ry - jp.ry;													// 1 FLOPs
	real_tn rz = ip.rz - jp.rz;													// 1 FLOPs
	real_tn e2 = ip.e2 + jp.e2;													// 1 FLOPs
	real_tn r2 = rx * rx + ry * ry + rz * rz;									// 5 FLOPs
	int_tn mask = (r2 > 0);

	real_tn m_r3 = smoothed_m_r3(jp.m, r2, e2, mask);							// 5 FLOPs

	ip.ax -= m_r3 * rx;															// 2 FLOPs
	ip.ay -= m_r3 * ry;															// 2 FLOPs
	ip.az -= m_r3 * rz;															// 2 FLOPs
	return ip;
}
// Total flop count: 20


#endif	// __ACC_KERNEL_COMMON_H__
