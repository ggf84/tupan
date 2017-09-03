#include "kepler_solver_kernel_common.h"


void
kepler_solver_kernel(
	const real_t dt,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	real_t __irx[],
	real_t __iry[],
	real_t __irz[],
	real_t __ivx[],
	real_t __ivy[],
	real_t __ivz[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	real_t __jrx[],
	real_t __jry[],
	real_t __jrz[],
	real_t __jvx[],
	real_t __jvy[],
	real_t __jvz[])
{
	kepler_solver_kernel_core(
		dt,
		__im[0], __irx[0], __iry[0], __irz[0],
		__ie2[0], __ivx[0], __ivy[0], __ivz[0],
		__jm[0], __jrx[0], __jry[0], __jrz[0],
		__je2[0], __jvx[0], __jvy[0], __jvz[0],
		__irx, __iry, __irz,
		__ivx, __ivy, __ivz,
		__jrx, __jry, __jrz,
		__jvx, __jvy, __jvz);
}

