#ifndef __NREG_KERNELS_COMMON_H__
#define __NREG_KERNELS_COMMON_H__

#include "common.h"
#include "smoothing.h"


#define NREG_X_DECL_STRUCTS(iT, jT)			\
	typedef struct nreg_x_idata {			\
		iT drx, dry, drz, ax, ay, az, u;	\
		iT rx, ry, rz, vx, vy, vz, e2, m;	\
	} Nreg_X_IData;							\
	typedef struct nreg_x_jdata {			\
		jT rx, ry, rz, vx, vy, vz, e2, m;	\
	} Nreg_X_JData;

NREG_X_DECL_STRUCTS(real_tn, real_t)


static inline Nreg_X_IData
nreg_Xkernel_core(Nreg_X_IData ip, Nreg_X_JData jp, const real_tn dt)
{
	real_tn rx = ip.rx - jp.rx;													// 1 FLOPs
	real_tn ry = ip.ry - jp.ry;													// 1 FLOPs
	real_tn rz = ip.rz - jp.rz;													// 1 FLOPs
	real_tn vx = ip.vx - jp.vx;													// 1 FLOPs
	real_tn vy = ip.vy - jp.vy;													// 1 FLOPs
	real_tn vz = ip.vz - jp.vz;													// 1 FLOPs
	real_tn e2 = ip.e2 + jp.e2;													// 1 FLOPs

	rx += vx * dt;																// 2 FLOPs
	ry += vy * dt;																// 2 FLOPs
	rz += vz * dt;																// 2 FLOPs

	real_tn r2 = rx * rx + ry * ry + rz * rz;									// 5 FLOPs
	int_tn mask = (r2 > 0);

	real_tn m_r3;
	real_tn m_r1 = smoothed_m_r1_m_r3(jp.m, r2, e2, mask, &m_r3);				// 5 FLOPs

	ip.drx += jp.m * rx;														// 2 FLOPs
	ip.dry += jp.m * ry;														// 2 FLOPs
	ip.drz += jp.m * rz;														// 2 FLOPs
	ip.ax -= m_r3 * rx;															// 2 FLOPs
	ip.ay -= m_r3 * ry;															// 2 FLOPs
	ip.az -= m_r3 * rz;															// 2 FLOPs
	ip.u += m_r1;																// 1 FLOPs
	return ip;
}
// Total flop count: 37


#define NREG_V_DECL_STRUCTS(iT, jT)		\
	typedef struct nreg_v_idata {		\
		iT dvx, dvy, dvz, k;			\
		iT vx, vy, vz, ax, ay, az, m;	\
	} Nreg_V_IData;						\
	typedef struct nreg_v_jdata {		\
		jT vx, vy, vz, ax, ay, az, m;	\
	} Nreg_V_JData;

NREG_V_DECL_STRUCTS(real_tn, real_t)


static inline Nreg_V_IData
nreg_Vkernel_core(Nreg_V_IData ip, Nreg_V_JData jp, const real_tn dt)
{
	real_tn vx = ip.vx - jp.vx;													// 1 FLOPs
	real_tn vy = ip.vy - jp.vy;													// 1 FLOPs
	real_tn vz = ip.vz - jp.vz;													// 1 FLOPs
	real_tn ax = ip.ax - jp.ax;													// 1 FLOPs
	real_tn ay = ip.ay - jp.ay;													// 1 FLOPs
	real_tn az = ip.az - jp.az;													// 1 FLOPs

	vx += ax * dt;																// 2 FLOPs
	vy += ay * dt;																// 2 FLOPs
	vz += az * dt;																// 2 FLOPs

	real_tn v2 = vx * vx + vy * vy + vz * vz;									// 5 FLOPs

	ip.dvx += jp.m * vx;														// 2 FLOPs
	ip.dvy += jp.m * vy;														// 2 FLOPs
	ip.dvz += jp.m * vz;														// 2 FLOPs
	ip.k += jp.m * v2;															// 2 FLOPs
	return ip;
}
// Total flop count: 25


#endif	// __NREG_KERNELS_COMMON_H__
