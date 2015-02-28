#ifndef __PNACC_KERNEL_COMMON_H__
#define __PNACC_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"
#include "pn_terms.h"


static inline void
pnacc_kernel_core(
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
	CLIGHT const clight,
	real_tn *ipnax,
	real_tn *ipnay,
	real_tn *ipnaz)
{
	real_tn rx = irx - jrx;														// 1 FLOPs
	real_tn ry = iry - jry;														// 1 FLOPs
	real_tn rz = irz - jrz;														// 1 FLOPs
	real_tn e2 = ie2 + je2;														// 1 FLOPs
	real_tn vx = ivx - jvx;														// 1 FLOPs
	real_tn vy = ivy - jvy;														// 1 FLOPs
	real_tn vz = ivz - jvz;														// 1 FLOPs
//	real_tn m = im + jm;														// 1 FLOPs
	real_tn r2 = rx * rx + ry * ry + rz * rz;									// 5 FLOPs
	real_tn v2 = vx * vx + vy * vy + vz * vz;									// 5 FLOPs
	int_tn mask = (r2 > 0);

	real_tn inv_r2;
	real_tn inv_r1 = smoothed_inv_r1r2(r2, e2, mask, &inv_r2);					// 3 FLOPs

	real_tn nx = rx * inv_r1;													// 1 FLOPs
	real_tn ny = ry * inv_r1;													// 1 FLOPs
	real_tn nz = rz * inv_r1;													// 1 FLOPs

//	real_tn r_sch = 2 * m * clight.inv2;										// 2 FLOPs
//	real_tn gamma2_a = r_sch * inv_r1;										  	// 1 FLOPs
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
			im, jm,
			nx, ny, nz, vx, vy, vz, v2,
			ivx, ivy, ivz, jvx, jvy, jvz,
			inv_r1, inv_r2, clight, &pn);										// ??? FLOPs
//		pn.a = select((real_tn)(0), pn.a, mask);
//		pn.b = select((real_tn)(0), pn.b, mask);
//	}
	*ipnax += pn.a * nx + pn.b * vx;											// 4 FLOPs
	*ipnay += pn.a * ny + pn.b * vy;											// 4 FLOPs
	*ipnaz += pn.a * nz + pn.b * vz;											// 4 FLOPs
}
// Total flop count: 40+???


#endif	// __PNACC_KERNEL_COMMON_H__
