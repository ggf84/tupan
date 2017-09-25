#include "nbody_parallel.h"
#include "sakura_kernel_common.h"


void
sakura_kernel_rectangle(
	const real_t dt,
	const int_t flag,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	const real_t __ivel[],
	real_t __idpos[],
	real_t __idvel[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jpos[],
	const real_t __jvel[],
	real_t __jdpos[],
	real_t __jdvel[])
{
	using namespace Sakura;
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __ipos, __ivel);
	auto jpart = setup<tile>(nj, __jm, __je2, __jpos, __jvel);

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		P2P_sakura_kernel_core<tile>(dt, flag)
	);

	commit<tile>(ni, ipart, __idpos, __idvel);
	commit<tile>(nj, jpart, __jdpos, __jdvel);
}


void
sakura_kernel_triangle(
	const real_t dt,
	const int_t flag,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	const real_t __ivel[],
	real_t __idpos[],
	real_t __idvel[])
{
	using namespace Sakura;
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __ipos, __ivel);

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		P2P_sakura_kernel_core<tile>(dt, flag)
	);

	commit<tile>(ni, ipart, __idpos, __idvel);
}

