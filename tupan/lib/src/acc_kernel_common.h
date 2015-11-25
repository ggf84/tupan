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


static inline void
p2p_acc_kernel_core(Acc_Data * restrict ip, Acc_Data * restrict jp)
{
	real_tn rx = ip->rx - jp->rx;
	real_tn ry = ip->ry - jp->ry;
	real_tn rz = ip->rz - jp->rz;
	real_tn e2 = ip->e2 + jp->e2;
	real_tn r2 = rx * rx + ry * ry + rz * rz;
	real_tn inv_r3 = smoothed_inv_r3(r2, e2);	// 5 FLOPs
	{	// i-part
		real_tn m_r3 = jp->m * inv_r3;
		ip->ax -= m_r3 * rx;
		ip->ay -= m_r3 * ry;
		ip->az -= m_r3 * rz;
	}
	{	// j-part
		real_tn m_r3 = ip->m * inv_r3;
		jp->ax += m_r3 * rx;
		jp->ay += m_r3 * ry;
		jp->az += m_r3 * rz;
	}
}
// Total flop count: 28

#endif	// __ACC_KERNEL_COMMON_H__
