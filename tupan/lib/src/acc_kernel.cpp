#include "nbody_parallel.h"
#include "acc_kernel_common.h"


void
acc_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	real_t __iacc[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jpos[],
	real_t __jacc[])
{
	using namespace Acc;
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __ipos);
	auto jpart = setup<tile>(nj, __jm, __je2, __jpos);

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		P2P_acc_kernel_core<tile>()
	);

	commit<tile>(ni, ipart, __iacc);
	commit<tile>(nj, jpart, __jacc);
}


void
acc_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	real_t __iacc[])
{
	using namespace Acc;
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __ipos);

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		P2P_acc_kernel_core<tile>()
	);

	commit<tile>(ni, ipart, __iacc);
}

