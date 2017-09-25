#include "nbody_parallel.h"
#include "pnacc_kernel_common.h"


void
pnacc_kernel_rectangle(
	const uint_t order,
	const real_t clight,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	const real_t __ivel[],
	real_t __ipnacc[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jpos[],
	const real_t __jvel[],
	real_t __jpnacc[])
{
	using namespace PNAcc;
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __ipos, __ivel);
	auto jpart = setup<tile>(nj, __jm, __je2, __jpos, __jvel);

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		P2P_pnacc_kernel_core<tile>(order, clight)
	);

	commit<tile>(ni, ipart, __ipnacc);
	commit<tile>(nj, jpart, __jpnacc);
}


void
pnacc_kernel_triangle(
	const uint_t order,
	const real_t clight,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	const real_t __ivel[],
	real_t __ipnacc[])
{
	using namespace PNAcc;
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __ipos, __ivel);

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		P2P_pnacc_kernel_core<tile>(order, clight)
	);

	commit<tile>(ni, ipart, __ipnacc);
}

