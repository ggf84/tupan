#include "nbody_parallel.h"
#include "phi_kernel_common.h"


void
phi_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __iphi[],
	real_t __jphi[])
{
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __irdot);
	auto jpart = setup<tile>(nj, __jm, __je2, __jrdot);

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		P2P_phi_kernel_core<tile>()
	);

	commit<tile>(ni, ipart, __iphi);
	commit<tile>(nj, jpart, __jphi);
}


void
phi_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iphi[])
{
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __irdot);

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		P2P_phi_kernel_core<tile>()
	);

	commit<tile>(ni, ipart, __iphi);
}


void
phi_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __iphi[])
{
	for (uint_t i = 0; i < ni; ++i) {
		Phi_Data ip;
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (auto kdot = 0; kdot < 1; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
			}
		}
		ip.phi = 0;

		for (uint_t j = 0; j < nj; ++j) {
			Phi_Data jp;
			jp.m = __jm[j];
			jp.e2 = __je2[j];
			for (auto kdot = 0; kdot < 1; ++kdot) {
				for (auto kdim = 0; kdim < NDIM; ++kdim) {
					const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
					jp.rdot[kdot][kdim] = ptr[j];
				}
			}
			jp.phi = 0;
			ip = phi_kernel_core(ip, jp);
		}

		__iphi[i] = ip.phi;
	}
}

