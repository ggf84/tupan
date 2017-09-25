#include "nbody_parallel.h"
#include "snp_crk_kernel_common.h"


void
snp_crk_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	const real_t __ivel[],
	const real_t __iacc[],
	const real_t __ijrk[],
	real_t __if0[],
	real_t __if1[],
	real_t __if2[],
	real_t __if3[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jpos[],
	const real_t __jvel[],
	const real_t __jacc[],
	const real_t __jjrk[],
	real_t __jf0[],
	real_t __jf1[],
	real_t __jf2[],
	real_t __jf3[])
{
	using namespace Snp_Crk;
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __ipos, __ivel, __iacc, __ijrk);
	auto jpart = setup<tile>(nj, __jm, __je2, __jpos, __jvel, __jacc, __jjrk);

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		P2P_snp_crk_kernel_core<tile>()
	);

	commit<tile>(ni, ipart, __if0, __if1, __if2, __if3);
	commit<tile>(nj, jpart, __jf0, __jf1, __jf2, __jf3);
}


void
snp_crk_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	const real_t __ivel[],
	const real_t __iacc[],
	const real_t __ijrk[],
	real_t __if0[],
	real_t __if1[],
	real_t __if2[],
	real_t __if3[])
{
	using namespace Snp_Crk;
	constexpr auto tile = 64 / sizeof(real_t);

	auto ipart = setup<tile>(ni, __im, __ie2, __ipos, __ivel, __iacc, __ijrk);

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		P2P_snp_crk_kernel_core<tile>()
	);

	commit<tile>(ni, ipart, __if0, __if1, __if2, __if3);
}

