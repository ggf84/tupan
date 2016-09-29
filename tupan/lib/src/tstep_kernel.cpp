#include "nbody_parallel.h"
#include "tstep_kernel_common.h"


void
tstep_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	const real_t eta,
	real_t __idt_a[],
	real_t __idt_b[],
	real_t __jdt_a[],
	real_t __jdt_b[])
{
	constexpr auto tile = 16;

	auto ipart = setup<tile>(ni, __im, __ie2, __irdot);
	auto jpart = setup<tile>(nj, __jm, __je2, __jrdot);

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		P2P_tstep_kernel_core<tile>(eta)
	);

	commit<tile>(ni, ipart, __idt_a, __idt_b, eta);
	commit<tile>(nj, jpart, __jdt_a, __jdt_b, eta);
}


void
tstep_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const real_t eta,
	real_t __idt_a[],
	real_t __idt_b[])
{
	constexpr auto tile = 16;

	auto ipart = setup<tile>(ni, __im, __ie2, __irdot);

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		P2P_tstep_kernel_core<tile>(eta)
	);

	commit<tile>(ni, ipart, __idt_a, __idt_b, eta);
}


void
tstep_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	const real_t eta,
	real_t __idt_a[],
	real_t __idt_b[])
{
	for (uint_t i = 0; i < ni; ++i) {
		Tstep_Data ip;
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (uint_t kdot = 0; kdot < 2; ++kdot) {
			for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
			}
			ip.w2[kdot] = 0;
		}

		for (uint_t j = 0; j < nj; ++j) {
			Tstep_Data jp;
			jp.m = __jm[j];
			jp.e2 = __je2[j];
			for (uint_t kdot = 0; kdot < 2; ++kdot) {
				for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
					const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
					jp.rdot[kdot][kdim] = ptr[j];
				}
				jp.w2[kdot] = 0;
			}
			ip = tstep_kernel_core(ip, jp, eta);
		}

		__idt_a[i] = eta / sqrt(ip.w2[0]);
		__idt_b[i] = eta / sqrt(ip.w2[1]);
	}
}

