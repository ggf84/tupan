#include "nbody_parallel.h"
#include "phi_kernel_common.h"


void
phi_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	real_t __iphi[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jpos[],
	real_t __jphi[])
{
	using namespace Phi;
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __ipos);
	auto jpart = setup<tile>(nj, __jm, __je2, __jpos);

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
	const real_t __ipos[],
	real_t __iphi[])
{
	using namespace Phi;
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __ipos);

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		P2P_phi_kernel_core<tile>()
	);

	commit<tile>(ni, ipart, __iphi);
}

