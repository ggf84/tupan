#include "pnacc_kernel_common.h"


void
pnacc_kernel(
	uint_t const ni,
	real_t const __im[restrict],
	real_t const __irx[restrict],
	real_t const __iry[restrict],
	real_t const __irz[restrict],
	real_t const __ie2[restrict],
	real_t const __ivx[restrict],
	real_t const __ivy[restrict],
	real_t const __ivz[restrict],
	uint_t const nj,
	real_t const __jm[restrict],
	real_t const __jrx[restrict],
	real_t const __jry[restrict],
	real_t const __jrz[restrict],
	real_t const __je2[restrict],
	real_t const __jvx[restrict],
	real_t const __jvy[restrict],
	real_t const __jvz[restrict],
	CLIGHT const clight,
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

