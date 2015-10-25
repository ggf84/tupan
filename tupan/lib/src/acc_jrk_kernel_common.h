#ifndef __ACC_JRK_KERNEL_COMMON_H__
#define __ACC_JRK_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#define ACC_JRK_DECL_STRUCTS(iT, jT)		\
	typedef struct acc_jrk_idata {			\
		iT ax, ay, az, jx, jy, jz;			\
		iT rx, ry, rz, vx, vy, vz, e2, m;	\
	} Acc_Jrk_IData;						\
	typedef struct acc_jrk_jdata {			\
		jT rx, ry, rz, vx, vy, vz, e2, m;	\
	} Acc_Jrk_JData;

ACC_JRK_DECL_STRUCTS(real_tn, real_t)


static inline Acc_Jrk_IData
acc_jrk_kernel_core(Acc_Jrk_IData ip, Acc_Jrk_JData jp)
{
	real_tn rx = ip.rx - jp.rx;													// 1 FLOPs
	real_tn ry = ip.ry - jp.ry;													// 1 FLOPs
	real_tn rz = ip.rz - jp.rz;													// 1 FLOPs
	real_tn vx = ip.vx - jp.vx;													// 1 FLOPs
	real_tn vy = ip.vy - jp.vy;													// 1 FLOPs
	real_tn vz = ip.vz - jp.vz;													// 1 FLOPs
	real_tn e2 = ip.e2 + jp.e2;													// 1 FLOPs
	real_tn r2 = rx * rx + ry * ry + rz * rz;									// 5 FLOPs
	real_tn rv = rx * vx + ry * vy + rz * vz;									// 5 FLOPs
	int_tn mask = (r2 > 0);

	real_tn inv_r2;
	real_tn m_r3 = smoothed_m_r3_inv_r2(jp.m, r2, e2, mask, &inv_r2);			// 5 FLOPs

	real_tn alpha = 3 * rv * inv_r2;											// 2 FLOPs

	vx -= alpha * rx;															// 2 FLOPs
	vy -= alpha * ry;															// 2 FLOPs
	vz -= alpha * rz;															// 2 FLOPs

	ip.ax -= m_r3 * rx;															// 2 FLOPs
	ip.ay -= m_r3 * ry;															// 2 FLOPs
	ip.az -= m_r3 * rz;															// 2 FLOPs
	ip.jx -= m_r3 * vx;															// 2 FLOPs
	ip.jy -= m_r3 * vy;															// 2 FLOPs
	ip.jz -= m_r3 * vz;															// 2 FLOPs
	return ip;
}
// Total flop count: 42


#endif	// __ACC_JRK_KERNEL_COMMON_H__
