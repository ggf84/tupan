#include "pnacc_kernel_common.h"


void
pnacc_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __jrx[],
	const real_t __jry[],
	const real_t __jrz[],
	const real_t __je2[],
	const real_t __jvx[],
	const real_t __jvy[],
	const real_t __jvz[],
	const CLIGHT clight,
	real_t __ipnax[],
	real_t __ipnay[],
	real_t __ipnaz[])
{
	for (uint_t i = 0; i < ni; ++i) {
		PNAcc_Data ip = (PNAcc_Data){
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
			PNAcc_Data jp = (PNAcc_Data){
				.pnax = 0,
				.pnay = 0,
				.pnaz = 0,
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

