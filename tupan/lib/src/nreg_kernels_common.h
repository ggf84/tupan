#ifndef __NREG_KERNELS_COMMON_H__
#define __NREG_KERNELS_COMMON_H__

#include "common.h"
#include "smoothing.h"


static inline void
nreg_Xkernel_core(
	real_tn const dt,
	real_tn const im,
	real_tn const irx,
	real_tn const iry,
	real_tn const irz,
	real_tn const ie2,
	real_tn const ivx,
	real_tn const ivy,
	real_tn const ivz,
	real_tn const jm,
	real_tn const jrx,
	real_tn const jry,
	real_tn const jrz,
	real_tn const je2,
	real_tn const jvx,
	real_tn const jvy,
	real_tn const jvz,
	real_tn *idrx,
	real_tn *idry,
	real_tn *idrz,
	real_tn *iax,
	real_tn *iay,
	real_tn *iaz,
	real_tn *iu)
{
	real_tn rx = irx - jrx;														// 1 FLOPs
	real_tn ry = iry - jry;														// 1 FLOPs
	real_tn rz = irz - jrz;														// 1 FLOPs
	real_tn e2 = ie2 + je2;														// 1 FLOPs
	real_tn vx = ivx - jvx;														// 1 FLOPs
	real_tn vy = ivy - jvy;														// 1 FLOPs
	real_tn vz = ivz - jvz;														// 1 FLOPs

	rx += vx * dt;																// 2 FLOPs
	ry += vy * dt;																// 2 FLOPs
	rz += vz * dt;																// 2 FLOPs

	real_tn r2 = rx * rx + ry * ry + rz * rz;									// 5 FLOPs
	int_tn mask = (r2 > 0);

	real_tn m_r3;
	real_tn m_r1 = smoothed_m_r1_m_r3(jm, r2, e2, mask, &m_r3);					// 5 FLOPs

	*idrx += jm * rx;															// 2 FLOPs
	*idry += jm * ry;															// 2 FLOPs
	*idrz += jm * rz;															// 2 FLOPs
	*iax -= m_r3 * rx;															// 2 FLOPs
	*iay -= m_r3 * ry;															// 2 FLOPs
	*iaz -= m_r3 * rz;															// 2 FLOPs
	*iu += m_r1;																// 1 FLOPs
}
// Total flop count: 37


static inline void
nreg_Vkernel_core(
	real_tn const dt,
	real_tn const im,
	real_tn const ivx,
	real_tn const ivy,
	real_tn const ivz,
	real_tn const iax,
	real_tn const iay,
	real_tn const iaz,
	real_tn const jm,
	real_tn const jvx,
	real_tn const jvy,
	real_tn const jvz,
	real_tn const jax,
	real_tn const jay,
	real_tn const jaz,
	real_tn *idvx,
	real_tn *idvy,
	real_tn *idvz,
	real_tn *ik)
{
	real_tn vx = ivx - jvx;														// 1 FLOPs
	real_tn vy = ivy - jvy;														// 1 FLOPs
	real_tn vz = ivz - jvz;														// 1 FLOPs
	real_tn ax = iax - jax;														// 1 FLOPs
	real_tn ay = iay - jay;														// 1 FLOPs
	real_tn az = iaz - jaz;														// 1 FLOPs

	vx += ax * dt;																// 2 FLOPs
	vy += ay * dt;																// 2 FLOPs
	vz += az * dt;																// 2 FLOPs

	real_tn v2 = vx * vx + vy * vy + vz * vz;									// 5 FLOPs

	*idvx += jm * vx;															// 2 FLOPs
	*idvy += jm * vy;															// 2 FLOPs
	*idvz += jm * vz;															// 2 FLOPs
	*ik += jm * v2;																// 2 FLOPs
}
// Total flop count: 25


#endif	// __NREG_KERNELS_COMMON_H__
