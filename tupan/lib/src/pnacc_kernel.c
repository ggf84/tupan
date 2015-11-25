#include "pnacc_kernel_common.h"


void
pnacc_kernel(
	const uint_t ni,
	const real_t __im[restrict],
	const real_t __irx[restrict],
	const real_t __iry[restrict],
	const real_t __irz[restrict],
	const real_t __ie2[restrict],
	const real_t __ivx[restrict],
	const real_t __ivy[restrict],
	const real_t __ivz[restrict],
	const uint_t nj,
	const real_t __jm[restrict],
	const real_t __jrx[restrict],
	const real_t __jry[restrict],
	const real_t __jrz[restrict],
	const real_t __je2[restrict],
	const real_t __jvx[restrict],
	const real_t __jvy[restrict],
	const real_t __jvz[restrict],
	const CLIGHT clight,
	real_t __ipnax[restrict],
	real_t __ipnay[restrict],
	real_t __ipnaz[restrict])
{
	for (uint_t i = 0; i < ni; ++i) {
		PNAcc_IData ip = (PNAcc_IData){
			.pnax = 0,
			.pnay = 0,
			.pnaz = 0,
			.rx = __irx[i],
			.ry = __iry[i],
			.rz = __irz[i],
			.vx = __ivx[i],
			.vy = __ivy[i],
			.vz = __ivz[i],
			.e2 = __ie2[i],
			.m = __im[i],
		};

		for (uint_t j = 0; j < nj; ++j) {
			PNAcc_JData jp = (PNAcc_JData){
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.e2 = __je2[j],
				.m = __jm[j],
			};
			ip = pnacc_kernel_core(ip, jp, clight);
		}

		__ipnax[i] = ip.pnax;
		__ipnay[i] = ip.pnay;
		__ipnaz[i] = ip.pnaz;
	}
}

