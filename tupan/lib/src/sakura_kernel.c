#include "sakura_kernel_common.h"


void
sakura_kernel(
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
	real_t const dt,
	int_t const flag,
	real_t __idrx[restrict],
	real_t __idry[restrict],
	real_t __idrz[restrict],
	real_t __idvx[restrict],
	real_t __idvy[restrict],
	real_t __idvz[restrict])
{
	for (uint_t i = 0; i < ni; ++i) {
		Sakura_IData ip = (Sakura_IData){
			.drx = 0,
			.dry = 0,
			.drz = 0,
			.dvx = 0,
			.dvy = 0,
			.dvz = 0,
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
			Sakura_JData jp = (Sakura_JData){
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.e2 = __je2[j],
				.m = __jm[j],
			};
			ip = sakura_kernel_core(ip, jp, dt, flag);
		}

		__idrx[i] = ip.drx;
		__idry[i] = ip.dry;
		__idrz[i] = ip.drz;
		__idvx[i] = ip.dvx;
		__idvy[i] = ip.dvy;
		__idvz[i] = ip.dvz;
	}
}

