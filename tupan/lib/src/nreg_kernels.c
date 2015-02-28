#include "nreg_kernels_common.h"


void
nreg_Xkernel(
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
	real_t __idrx[restrict],
	real_t __idry[restrict],
	real_t __idrz[restrict],
	real_t __iax[restrict],
	real_t __iay[restrict],
	real_t __iaz[restrict],
	real_t __iu[restrict])
{
	for (uint_t i = 0; i < ni; ++i) {
		real_t im = __im[i];
		real_t irx = __irx[i];
		real_t iry = __iry[i];
		real_t irz = __irz[i];
		real_t ie2 = __ie2[i];
		real_t ivx = __ivx[i];
		real_t ivy = __ivy[i];
		real_t ivz = __ivz[i];
		real_t idrx = 0;
		real_t idry = 0;
		real_t idrz = 0;
		real_t iax = 0;
		real_t iay = 0;
		real_t iaz = 0;
		real_t iu = 0;

		for (uint_t j = 0; j < nj; ++j) {
			real_t jm = __jm[j];
			real_t jrx = __jrx[j];
			real_t jry = __jry[j];
			real_t jrz = __jrz[j];
			real_t je2 = __je2[j];
			real_t jvx = __jvx[j];
			real_t jvy = __jvy[j];
			real_t jvz = __jvz[j];
			nreg_Xkernel_core(
				dt,
				im, irx, iry, irz, ie2, ivx, ivy, ivz,
				jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
				&idrx, &idry, &idrz, &iax, &iay, &iaz, &iu);
		}

		__idrx[i] = idrx;
		__idry[i] = idry;
		__idrz[i] = idrz;
		__iax[i] = iax;
		__iay[i] = iay;
		__iaz[i] = iaz;
		__iu[i] = im * iu;
	}
}


void
nreg_Vkernel(
	uint_t const ni,
	real_t const __im[restrict],
	real_t const __ivx[restrict],
	real_t const __ivy[restrict],
	real_t const __ivz[restrict],
	real_t const __iax[restrict],
	real_t const __iay[restrict],
	real_t const __iaz[restrict],
	uint_t const nj,
	real_t const __jm[restrict],
	real_t const __jvx[restrict],
	real_t const __jvy[restrict],
	real_t const __jvz[restrict],
	real_t const __jax[restrict],
	real_t const __jay[restrict],
	real_t const __jaz[restrict],
	real_t const dt,
	real_t __idvx[restrict],
	real_t __idvy[restrict],
	real_t __idvz[restrict],
	real_t __ik[restrict])
{
	for (uint_t i = 0; i < ni; ++i) {
		real_t im = __im[i];
		real_t ivx = __ivx[i];
		real_t ivy = __ivy[i];
		real_t ivz = __ivz[i];
		real_t iax = __iax[i];
		real_t iay = __iay[i];
		real_t iaz = __iaz[i];
		real_t idvx = 0;
		real_t idvy = 0;
		real_t idvz = 0;
		real_t ik = 0;

		for (uint_t j = 0; j < nj; ++j) {
			real_t jm = __jm[j];
			real_t jvx = __jvx[j];
			real_t jvy = __jvy[j];
			real_t jvz = __jvz[j];
			real_t jax = __jax[j];
			real_t jay = __jay[j];
			real_t jaz = __jaz[j];
			nreg_Vkernel_core(
				dt,
				im, ivx, ivy, ivz, iax, iay, iaz,
				jm, jvx, jvy, jvz, jax, jay, jaz,
				&idvx, &idvy, &idvz, &ik);
		}

		__idvx[i] = idvx;
		__idvy[i] = idvy;
		__idvz[i] = idvz;
		__ik[i] = im * ik;
	}
}

