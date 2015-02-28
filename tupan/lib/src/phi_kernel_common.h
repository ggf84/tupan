#ifndef __PHI_KERNEL_COMMON_H__
#define __PHI_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"


static inline void
phi_kernel_core(
	real_tn const im,
	real_tn const irx,
	real_tn const iry,
	real_tn const irz,
	real_tn const ie2,
	real_tn const jm,
	real_tn const jrx,
	real_tn const jry,
	real_tn const jrz,
	real_tn const je2,
	real_tn *iphi)
{
	real_tn rx = irx - jrx;														// 1 FLOPs
	real_tn ry = iry - jry;														// 1 FLOPs
	real_tn rz = irz - jrz;														// 1 FLOPs
	real_tn e2 = ie2 + je2;														// 1 FLOPs
	real_tn r2 = rx * rx + ry * ry + rz * rz;									// 5 FLOPs
	int_tn mask = (r2 > 0);

	real_tn m_r1 = smoothed_m_r1(jm, r2, e2, mask);								// 4 FLOPs

	*iphi -= m_r1;																// 1 FLOPs
}
// Total flop count: 14


#endif	// __PHI_KERNEL_COMMON_H__
