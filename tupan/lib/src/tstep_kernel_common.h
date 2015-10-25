#ifndef __TSTEP_KERNEL_COMMON_H__
#define __TSTEP_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


#define TSTEP_DECL_STRUCTS(iT, jT)			\
	typedef struct tstep_idata {			\
		iT w2_a, w2_b;						\
		iT rx, ry, rz, vx, vy, vz, e2, m;	\
	} Tstep_IData;							\
	typedef struct tstep_jdata {			\
		jT rx, ry, rz, vx, vy, vz, e2, m;	\
	} Tstep_JData;

TSTEP_DECL_STRUCTS(real_tn, real_t)


static inline Tstep_IData
tstep_kernel_core(Tstep_IData ip, Tstep_JData jp, real_t const eta)
{
	real_tn rx = ip.rx - jp.rx;													// 1 FLOPs
	real_tn ry = ip.ry - jp.ry;													// 1 FLOPs
	real_tn rz = ip.rz - jp.rz;													// 1 FLOPs
	real_tn vx = ip.vx - jp.vx;													// 1 FLOPs
	real_tn vy = ip.vy - jp.vy;													// 1 FLOPs
	real_tn vz = ip.vz - jp.vz;													// 1 FLOPs
	real_tn e2 = ip.e2 + jp.e2;													// 1 FLOPs
	real_tn m = ip.m + jp.m;													// 1 FLOPs
	real_tn r2 = rx * rx + ry * ry + rz * rz;									// 5 FLOPs
	real_tn rv = rx * vx + ry * vy + rz * vz;									// 5 FLOPs
	real_tn v2 = vx * vx + vy * vy + vz * vz;									// 5 FLOPs
	int_tn mask = (r2 > 0);

	real_tn inv_r2;
	real_tn m_r1 = smoothed_m_r1_inv_r2(m, r2, e2, mask, &inv_r2);				// 4 FLOPs

	real_tn a = (real_tn)(2);
	real_tn b = (1 + a / 2) * inv_r2;											// 3 FLOPs

	real_tn w2 = (v2 + a * m_r1) * inv_r2;										// 3 FLOPs
	real_tn gamma = (w2 + b * m_r1) * inv_r2;									// 3 FLOPs
	gamma *= (eta * rsqrt(w2));													// 3 FLOPs
	gamma = select((real_tn)(0), gamma, mask);
	w2 -= gamma * rv;															// 2 FLOPs

	ip.w2_a += w2;																// 1 FLOPs
	ip.w2_b = fmax(w2, ip.w2_b);
	return ip;
}
// Total flop count: 42


#endif	// __TSTEP_KERNEL_COMMON_H__
