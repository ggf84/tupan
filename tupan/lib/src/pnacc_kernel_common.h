#ifndef __PNACC_KERNEL_COMMON_H__
#define __PNACC_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"
#include "pn_terms.h"


#define PNACC_DECL_STRUCTS(iT, jT)			\
	typedef struct pnacc_idata {			\
		iT pnax, pnay, pnaz;				\
		iT rx, ry, rz, vx, vy, vz, e2, m;	\
	} PNAcc_IData;							\
	typedef struct pnacc_jdata {			\
		jT rx, ry, rz, vx, vy, vz, e2, m;	\
	} PNAcc_JData;

PNACC_DECL_STRUCTS(real_tn, real_t)


static inline PNAcc_IData
pnacc_kernel_core(PNAcc_IData ip, PNAcc_JData jp, CLIGHT const clight)
{
	real_tn rx = ip.rx - jp.rx;													// 1 FLOPs
	real_tn ry = ip.ry - jp.ry;													// 1 FLOPs
	real_tn rz = ip.rz - jp.rz;													// 1 FLOPs
	real_tn vx = ip.vx - jp.vx;													// 1 FLOPs
	real_tn vy = ip.vy - jp.vy;													// 1 FLOPs
	real_tn vz = ip.vz - jp.vz;													// 1 FLOPs
	real_tn e2 = ip.e2 + jp.e2;													// 1 FLOPs
//	real_tn m = ip.m + jp.m;													// 1 FLOPs
	real_tn r2 = rx * rx + ry * ry + rz * rz;									// 5 FLOPs
	real_tn v2 = vx * vx + vy * vy + vz * vz;									// 5 FLOPs
	int_tn mask = (r2 > 0);

	real_tn inv_r2;
	real_tn inv_r1 = smoothed_inv_r1r2(r2, e2, mask, &inv_r2);					// 3 FLOPs

	real_tn nx = rx * inv_r1;													// 1 FLOPs
	real_tn ny = ry * inv_r1;													// 1 FLOPs
	real_tn nz = rz * inv_r1;													// 1 FLOPs

//	real_tn r_sch = 2 * m * clight.inv2;										// 2 FLOPs
//	real_tn gamma2_a = r_sch * inv_r1;											// 1 FLOPs
//	real_tn gamma2_b = v2 * clight.inv2;										// 1 FLOPs
//	real_tn gamma2 = gamma2_a + gamma2_b;										// 1 FLOPs

	PN pn = PN_Init(0, 0);
	/* PN acceleration will only be calculated
	 * if the condition below is fulfilled:
	 * since gamma ~ v/c > 0.1% = 1e-3 therefore
	 * gamma2 > 1e-6 should be our condition.
	 */
//	int_tn mask = (gamma2 > (real_tn)(1.0e-6));
//	if (any(mask)) {
		p2p_pnterms(
			ip.m, jp.m,
			nx, ny, nz, vx, vy, vz, v2,
			ip.vx, ip.vy, ip.vz, jp.vx, jp.vy, jp.vz,
			inv_r1, inv_r2, clight, &pn);										// ??? FLOPs
//		pn.a = select((real_tn)(0), pn.a, mask);
//		pn.b = select((real_tn)(0), pn.b, mask);
//	}
	ip.pnax += pn.a * nx + pn.b * vx;											// 4 FLOPs
	ip.pnay += pn.a * ny + pn.b * vy;											// 4 FLOPs
	ip.pnaz += pn.a * nz + pn.b * vz;											// 4 FLOPs
	return ip;
}
// Total flop count: 40+???


#endif	// __PNACC_KERNEL_COMMON_H__
