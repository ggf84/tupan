#include "kepler_solver_kernel_common.h"


void
kepler_solver_kernel(
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
	real_t __ir1x[restrict],
	real_t __ir1y[restrict],
	real_t __ir1z[restrict],
	real_t __iv1x[restrict],
	real_t __iv1y[restrict],
	real_t __iv1z[restrict],
	real_t __jr1x[restrict],
	real_t __jr1y[restrict],
	real_t __jr1z[restrict],
	real_t __jv1x[restrict],
	real_t __jv1y[restrict],
	real_t __jv1z[restrict])
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

