#include "nbody_parallel.h"
#include "tstep_kernel_common.h"


void
tstep_kernel_rectangle(
	const real_t eta,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iw2_a[],
	real_t __iw2_b[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __jw2_a[],
	real_t __jw2_b[])
{
	using namespace Tstep;
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

	commit<tile>(ni, ipart, __iw2_a, __iw2_b, eta);
	commit<tile>(nj, jpart, __jw2_a, __jw2_b, eta);
}


void
tstep_kernel_triangle(
	const real_t eta,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iw2_a[],
	real_t __iw2_b[])
{
	using namespace Tstep;
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __irdot);

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		P2P_tstep_kernel_core<tile>(eta)
	);

	commit<tile>(ni, ipart, __iw2_a, __iw2_b, eta);
}

