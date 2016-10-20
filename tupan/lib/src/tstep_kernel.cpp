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
	constexpr auto tile = 64 / sizeof(real_t);

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
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __irdot);

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		P2P_tstep_kernel_core<tile>(eta)
	);

	commit<tile>(ni, ipart, __idt_a, __idt_b, eta);
}

