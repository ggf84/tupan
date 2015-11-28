#include "tstep_kernel_common.h"


void
tstep_kernel(
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
	const real_t eta,
	real_t __idt_a[],
	real_t __idt_b[])
{
	for (uint_t i = 0; i < ni; ++i) {
		Tstep_IData ip = (Tstep_IData){
			.w2_a = 0,
			.w2_b = 0,
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
			Tstep_JData jp = (Tstep_JData){
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.e2 = __je2[j],
				.m = __jm[j],
			};
			ip = tstep_kernel_core(ip, jp, eta);
		}

		__idt_a[i] = eta / sqrt(fmax((real_tn)(1), ip.w2_a));
		__idt_b[i] = eta / sqrt(fmax((real_tn)(1), ip.w2_b));
	}
}

