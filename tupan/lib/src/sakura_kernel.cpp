#include "nbody_parallel.h"
#include "sakura_kernel_common.h"


void
sakura_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	const real_t dt,
	const int_t flag,
	real_t __idrdot[],
	real_t __jdrdot[])
{
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __irdot);
	auto jpart = setup<tile>(nj, __jm, __je2, __jrdot);

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		P2P_sakura_kernel_core<tile>(dt, flag)
	);

	commit<tile>(ni, ipart, __idrdot);
	commit<tile>(nj, jpart, __jdrdot);
}


void
sakura_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const real_t dt,
	const int_t flag,
	real_t __idrdot[])
{
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __irdot);

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		P2P_sakura_kernel_core<tile>(dt, flag)
	);

	commit<tile>(ni, ipart, __idrdot);
}


void
sakura_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	const real_t dt,
	const int_t flag,
	real_t __idrdot[])
{
	for (uint_t i = 0; i < ni; ++i) {
		Sakura_Data ip;
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
				ip.drdot[kdot][kdim] = 0;
			}
		}

		for (uint_t j = 0; j < nj; ++j) {
			Sakura_Data jp;
			jp.m = __jm[j];
			jp.e2 = __je2[j];
			for (auto kdot = 0; kdot < 2; ++kdot) {
				for (auto kdim = 0; kdim < NDIM; ++kdim) {
					const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
					jp.rdot[kdot][kdim] = ptr[j];
					jp.drdot[kdot][kdim] = 0;
				}
			}
			ip = sakura_kernel_core(ip, jp, dt, flag);
		}

		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				real_t *ptr = &__idrdot[(kdot*NDIM+kdim)*ni];
				ptr[i] = ip.drdot[kdot][kdim];
			}
		}
	}
}

