#ifndef __PNACC_KERNEL_COMMON_H__
#define __PNACC_KERNEL_COMMON_H__

#include "common.h"
#include "smoothing.h"
#include "pn_terms.h"


#define PNACC_IMPLEMENT_STRUCT(N)							\
	typedef struct concat(pnacc_data, N) {					\
		concat(real_t, N) pnax, pnay, pnaz;					\
		concat(real_t, N) rx, ry, rz, vx, vy, vz, e2, m;	\
	} concat(PNAcc_Data, N);

PNACC_IMPLEMENT_STRUCT(1)
#if SIMD > 1
PNACC_IMPLEMENT_STRUCT(SIMD)
#endif
typedef PNAcc_Data1 PNAcc_Data;


static inline vec(PNAcc_Data)
pnacc_kernel_core(vec(PNAcc_Data) ip, PNAcc_Data jp, const CLIGHT clight)
// flop count: 36+???
{
	vec(real_t) rx = ip.rx - jp.rx;
	vec(real_t) ry = ip.ry - jp.ry;
	vec(real_t) rz = ip.rz - jp.rz;
	vec(real_t) vx = ip.vx - jp.vx;
	vec(real_t) vy = ip.vy - jp.vy;
	vec(real_t) vz = ip.vz - jp.vz;
	vec(real_t) e2 = ip.e2 + jp.e2;
//	vec(real_t) m = ip.m + jp.m;
	vec(real_t) r2 = rx * rx + ry * ry + rz * rz;
	vec(real_t) v2 = vx * vx + vy * vy + vz * vz;

	vec(real_t) inv_r1;
	vec(real_t) inv_r2 = smoothed_inv_r2_inv_r1(r2, e2, &inv_r1);	// flop count: 4
	inv_r1 = select((vec(real_t))(0), inv_r1, (r2 > 0));
	inv_r2 = select((vec(real_t))(0), inv_r2, (r2 > 0));

	vec(real_t) nx = rx * inv_r1;
	vec(real_t) ny = ry * inv_r1;
	vec(real_t) nz = rz * inv_r1;

//	vec(real_t) r_sch = 2 * m * clight.inv2;
//	vec(real_t) gamma2_a = r_sch * inv_r1;
//	vec(real_t) gamma2_b = v2 * clight.inv2;
//	vec(real_t) gamma2 = gamma2_a + gamma2_b;

	PN pn = PN_Init(0, 0);
	/* PN acceleration will only be calculated
	 * if the condition below is fulfilled:
	 * since gamma ~ v/c > 0.1% = 1e-3 therefore
	 * gamma2 > 1e-6 should be our condition.
	 */
//	vec(int_t) mask = (gamma2 > (vec(real_t))(1.0e-6));
//	if (any(mask)) {
		p2p_pnterms(
			ip.m, jp.m,
			nx, ny, nz, vx, vy, vz, v2,
			ip.vx, ip.vy, ip.vz, jp.vx, jp.vy, jp.vz,
			inv_r1, inv_r2, clight, &pn);	// ??? FLOPs
//		pn.a = select((vec(real_t))(0), pn.a, mask);
//		pn.b = select((vec(real_t))(0), pn.b, mask);
//	}
	ip.pnax += pn.a * nx + pn.b * vx;
	ip.pnay += pn.a * ny + pn.b * vy;
	ip.pnaz += pn.a * nz + pn.b * vz;
	return ip;
}


#endif	// __PNACC_KERNEL_COMMON_H__
