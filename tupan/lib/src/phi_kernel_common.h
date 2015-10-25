#ifndef __PHI_KERNEL_COMMON_H__
#define __PHI_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#define PHI_DECL_STRUCTS(iT, jT)	\
	typedef struct phi_idata {		\
		iT phi;						\
		iT rx, ry, rz, e2, m;		\
	} Phi_IData;					\
	typedef struct phi_jdata {		\
		jT rx, ry, rz, e2, m;		\
	} Phi_JData;

PHI_DECL_STRUCTS(real_tn, real_t)


static inline Phi_IData
phi_kernel_core(Phi_IData ip, Phi_JData jp)
{
	real_tn rx = ip.rx - jp.rx;													// 1 FLOPs
	real_tn ry = ip.ry - jp.ry;													// 1 FLOPs
	real_tn rz = ip.rz - jp.rz;													// 1 FLOPs
	real_tn e2 = ip.e2 + jp.e2;													// 1 FLOPs
	real_tn r2 = rx * rx + ry * ry + rz * rz;									// 5 FLOPs
	int_tn mask = (r2 > 0);

	real_tn m_r1 = smoothed_m_r1(jp.m, r2, e2, mask);							// 4 FLOPs

	ip.phi -= m_r1;																// 1 FLOPs
	return ip;
}
// Total flop count: 14


#endif	// __PHI_KERNEL_COMMON_H__
