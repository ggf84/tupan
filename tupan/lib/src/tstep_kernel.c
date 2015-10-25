#include "tstep_kernel_common.h"


void
tstep_kernel(
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
	real_t const eta,
	real_t __idt_a[restrict],
	real_t __idt_b[restrict])
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

