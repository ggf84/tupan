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
	using namespace Sakura;
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
	using namespace Sakura;
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

