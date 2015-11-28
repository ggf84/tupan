#include "kepler_solver_kernel_common.h"


void
kepler_solver_kernel(
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
	real_t __ir1x[],
	real_t __ir1y[],
	real_t __ir1z[],
	real_t __iv1x[],
	real_t __iv1y[],
	real_t __iv1z[],
	real_t __jr1x[],
	real_t __jr1y[],
	real_t __jr1z[],
	real_t __jv1x[],
	real_t __jv1y[],
	real_t __jv1z[])
{
	kepler_solver_kernel_core(
		dt,
		__im[0], __irx[0], __iry[0], __irz[0],
		__ie2[0], __ivx[0], __ivy[0], __ivz[0],
		__jm[0], __jrx[0], __jry[0], __jrz[0],
		__je2[0], __jvx[0], __jvy[0], __jvz[0],
		__ir1x, __ir1y, __ir1z,
		__iv1x, __iv1y, __iv1z,
		__jr1x, __jr1y, __jr1z,
		__jv1x, __jv1y, __jv1z);
}

