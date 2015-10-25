#ifndef __SNP_CRK_KERNEL_COMMON_H__
#define __SNP_CRK_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"

#define _3_2 (3/(real_t)(2))


#define SNP_CRK_DECL_STRUCTS(iT, jT)								\
	typedef struct snp_crk_idata {									\
		iT sx, sy, sz, cx, cy, cz;									\
		iT rx, ry, rz, vx, vy, vz, ax, ay, az, jx, jy, jz, e2, m;	\
	} Snp_Crk_IData;												\
	typedef struct snp_crk_jdata {									\
		jT rx, ry, rz, vx, vy, vz, ax, ay, az, jx, jy, jz, e2, m;	\
	} Snp_Crk_JData;

SNP_CRK_DECL_STRUCTS(real_tn, real_t)


static inline Snp_Crk_IData
snp_crk_kernel_core(Snp_Crk_IData ip, Snp_Crk_JData jp)
{
	real_tn rx = ip.rx - jp.rx;													// 1 FLOPs
	real_tn ry = ip.ry - jp.ry;													// 1 FLOPs
	real_tn rz = ip.rz - jp.rz;													// 1 FLOPs
	real_tn vx = ip.vx - jp.vx;													// 1 FLOPs
	real_tn vy = ip.vy - jp.vy;													// 1 FLOPs
	real_tn vz = ip.vz - jp.vz;													// 1 FLOPs
	real_tn ax = ip.ax - jp.ax;													// 1 FLOPs
	real_tn ay = ip.ay - jp.ay;													// 1 FLOPs
	real_tn az = ip.az - jp.az;													// 1 FLOPs
	real_tn jx = ip.jx - jp.jx;													// 1 FLOPs
	real_tn jy = ip.jy - jp.jy;													// 1 FLOPs
	real_tn jz = ip.jz - jp.jz;													// 1 FLOPs
	real_tn e2 = ip.e2 + jp.e2;													// 1 FLOPs
	real_tn r2 = rx * rx + ry * ry + rz * rz;									// 5 FLOPs
	real_tn rv = rx * vx + ry * vy + rz * vz;									// 5 FLOPs
	real_tn v2 = vx * vx + vy * vy + vz * vz;									// 5 FLOPs
	real_tn va = vx * ax + vy * ay + vz * az;									// 5 FLOPs
	real_tn ra = rx * ax + ry * ay + rz * az;									// 5 FLOPs
	real_tn rj = rx * jx + ry * jy + rz * jz;									// 5 FLOPs
	int_tn mask = (r2 > 0);

	real_tn inv_r2;
	real_tn m_r3 = smoothed_m_r3_inv_r2(jp.m, r2, e2, mask, &inv_r2);			// 5 FLOPs

	real_tn alpha = rv * inv_r2;												// 1 FLOPs
	real_tn alpha2 = alpha * alpha;												// 1 FLOPs
	real_tn beta = 3 * ((v2 + ra) * inv_r2 + alpha2);							// 4 FLOPs
	real_tn gamma = (3 * va + rj) * inv_r2 + alpha * (beta - 4 * alpha2);		// 7 FLOPs

	alpha *= 3;																	// 1 FLOPs
	gamma *= 3;																	// 1 FLOPs

	vx -= alpha * rx;															// 2 FLOPs
	vy -= alpha * ry;															// 2 FLOPs
	vz -= alpha * rz;															// 2 FLOPs

	alpha *= 2;																	// 1 FLOPs
	ax -= alpha * vx;															// 2 FLOPs
	ay -= alpha * vy;															// 2 FLOPs
	az -= alpha * vz;															// 2 FLOPs
	ax -= beta * rx;															// 2 FLOPs
	ay -= beta * ry;															// 2 FLOPs
	az -= beta * rz;															// 2 FLOPs

	alpha *= _3_2;																// 1 FLOPs
	beta *= 3;																	// 1 FLOPs
	jx -= alpha * ax;															// 2 FLOPs
	jy -= alpha * ay;															// 2 FLOPs
	jz -= alpha * az;															// 2 FLOPs
	jx -= beta * vx;															// 2 FLOPs
	jy -= beta * vy;															// 2 FLOPs
	jz -= beta * vz;															// 2 FLOPs
	jx -= gamma * rx;															// 2 FLOPs
	jy -= gamma * ry;															// 2 FLOPs
	jz -= gamma * rz;															// 2 FLOPs

	ip.sx -= m_r3 * ax;															// 2 FLOPs
	ip.sy -= m_r3 * ay;															// 2 FLOPs
	ip.sz -= m_r3 * az;															// 2 FLOPs
	ip.cx -= m_r3 * jx;															// 2 FLOPs
	ip.cy -= m_r3 * jy;															// 2 FLOPs
	ip.cz -= m_r3 * jz;															// 2 FLOPs
	return ip;
}
// Total flop count: 114


#endif	// __SNP_CRK_KERNEL_COMMON_H__
