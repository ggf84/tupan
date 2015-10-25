#include "acc_jrk_kernel_common.h"


void
acc_jrk_kernel(
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
	real_t __iax[restrict],
	real_t __iay[restrict],
	real_t __iaz[restrict],
	real_t __ijx[restrict],
	real_t __ijy[restrict],
	real_t __ijz[restrict])
{
	for (uint_t i = 0; i < ni; ++i) {
		Acc_Jrk_IData ip = (Acc_Jrk_IData){
			.ax = 0,
			.ay = 0,
			.az = 0,
			.jx = 0,
			.jy = 0,
			.jz = 0,
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
			Acc_Jrk_JData jp = (Acc_Jrk_JData){
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.e2 = __je2[j],
				.m = __jm[j],
			};
			ip = acc_jrk_kernel_core(ip, jp);
		}

		__iax[i] = ip.ax;
		__iay[i] = ip.ay;
		__iaz[i] = ip.az;
		__ijx[i] = ip.jx;
		__ijy[i] = ip.jy;
		__ijz[i] = ip.jz;
	}
}

