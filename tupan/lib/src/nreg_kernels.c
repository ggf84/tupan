#include "nreg_kernels_common.h"


void
nreg_Xkernel(
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
	const real_t dt,
	real_t __idrx[restrict],
	real_t __idry[restrict],
	real_t __idrz[restrict],
	real_t __iax[restrict],
	real_t __iay[restrict],
	real_t __iaz[restrict],
	real_t __iu[restrict])
{
	for (uint_t i = 0; i < ni; ++i) {
		Nreg_X_IData ip = (Nreg_X_IData){
			.drx = 0,
			.dry = 0,
			.drz = 0,
			.ax = 0,
			.ay = 0,
			.az = 0,
			.u = 0,
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
			Nreg_X_JData jp = (Nreg_X_JData){
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.e2 = __je2[j],
				.m = __jm[j],
			};
			ip = nreg_Xkernel_core(ip, jp, dt);
		}

		__idrx[i] = ip.drx;
		__idry[i] = ip.dry;
		__idrz[i] = ip.drz;
		__iax[i] = ip.ax;
		__iay[i] = ip.ay;
		__iaz[i] = ip.az;
		__iu[i] = ip.m * ip.u;
	}
}


void
nreg_Vkernel(
	const uint_t ni,
	const real_t __im[restrict],
	const real_t __ivx[restrict],
	const real_t __ivy[restrict],
	const real_t __ivz[restrict],
	const real_t __iax[restrict],
	const real_t __iay[restrict],
	const real_t __iaz[restrict],
	const uint_t nj,
	const real_t __jm[restrict],
	const real_t __jvx[restrict],
	const real_t __jvy[restrict],
	const real_t __jvz[restrict],
	const real_t __jax[restrict],
	const real_t __jay[restrict],
	const real_t __jaz[restrict],
	const real_t dt,
	real_t __idvx[restrict],
	real_t __idvy[restrict],
	real_t __idvz[restrict],
	real_t __ik[restrict])
{
	for (uint_t i = 0; i < ni; ++i) {
		Nreg_V_IData ip = (Nreg_V_IData){
			.dvx = 0,
			.dvy = 0,
			.dvz = 0,
			.k = 0,
			.vx = __ivx[i],
			.vy = __ivy[i],
			.vz = __ivz[i],
			.ax = __iax[i],
			.ay = __iay[i],
			.az = __iaz[i],
			.m = __im[i],
		};

		for (uint_t j = 0; j < nj; ++j) {
			Nreg_V_JData jp = (Nreg_V_JData){
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.ax = __jax[j],
				.ay = __jay[j],
				.az = __jaz[j],
				.m = __jm[j],
			};
			ip = nreg_Vkernel_core(ip, jp, dt);
		}

		__idvx[i] = ip.dvx;
		__idvy[i] = ip.dvy;
		__idvz[i] = ip.dvz;
		__ik[i] = ip.m * ip.k;
	}
}

