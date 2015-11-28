#include "acc_jrk_kernel_common.h"


void
acc_jrk_kernel(
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
	real_t __iax[],
	real_t __iay[],
	real_t __iaz[],
	real_t __ijx[],
	real_t __ijy[],
	real_t __ijz[])
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

