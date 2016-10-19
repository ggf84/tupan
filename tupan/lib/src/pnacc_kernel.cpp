#include "nbody_parallel.h"
#include "pnacc_kernel_common.h"


void
pnacc_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	const CLIGHT clight,
	real_t __ipnacc[],
	real_t __jpnacc[])
{
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __irdot);
	auto jpart = setup<tile>(nj, __jm, __je2, __jrdot);

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		P2P_pnacc_kernel_core<tile>(clight)
	);

	commit<tile>(ni, ipart, __ipnacc);
	commit<tile>(nj, jpart, __jpnacc);
}


void
pnacc_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const CLIGHT clight,
	real_t __ipnacc[])
{
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __irdot);

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		P2P_pnacc_kernel_core<tile>(clight)
	);

	commit<tile>(ni, ipart, __ipnacc);
}


void
pnacc_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	const CLIGHT clight,
	real_t __ipnacc[])
{
	for (uint_t i = 0; i < ni; ++i) {
		PNAcc_Data ip;
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (uint_t kdot = 0; kdot < 2; ++kdot) {
			for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
			}
		}
		for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
			ip.pnacc[kdim] = 0;
		}

		for (uint_t j = 0; j < nj; ++j) {
			PNAcc_Data jp;
			jp.m = __jm[j];
			jp.e2 = __je2[j];
			for (uint_t kdot = 0; kdot < 2; ++kdot) {
				for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
					const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
					jp.rdot[kdot][kdim] = ptr[j];
				}
			}
			for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
				jp.pnacc[kdim] = 0;
			}
			ip = pnacc_kernel_core(ip, jp, clight);
		}

		for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
			real_t *ptr = &__ipnacc[kdim*ni];
			ptr[i] = ip.pnacc[kdim];
		}
	}
}

