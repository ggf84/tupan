#include "sakura_kernel_common.h"


void
sakura_kernel(
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
	const real_t dt,
	const int_t flag,
	real_t __idrx[],
	real_t __idry[],
	real_t __idrz[],
	real_t __idvx[],
	real_t __idvy[],
	real_t __idvz[])
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

