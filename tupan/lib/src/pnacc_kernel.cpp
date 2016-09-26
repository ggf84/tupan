#include "nbody_parallel.h"
#include "pnacc_kernel_common.h"


void
pnacc_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	const CLIGHT clight,
	real_t __ipnacc[],
	real_t __jpnacc[])
{
	constexpr auto tile = 1;

	auto isize = (ni + tile - 1) / tile;
	vector<PNAcc_Data_SoA<tile>> ipart(isize);
	for (size_t i = 0; i < ni; ++i) {
		auto ii = i%tile;
		auto& ip = ipart[i/tile];
		ip.m[ii] = __im[i];
		ip.e2[ii] = __ie2[i];
		ip.rx[ii] = __irdot[(0*NDIM+0)*ni + i];
		ip.ry[ii] = __irdot[(0*NDIM+1)*ni + i];
		ip.rz[ii] = __irdot[(0*NDIM+2)*ni + i];
		ip.vx[ii] = __irdot[(1*NDIM+0)*ni + i];
		ip.vy[ii] = __irdot[(1*NDIM+1)*ni + i];
		ip.vz[ii] = __irdot[(1*NDIM+2)*ni + i];
		ip.pnax[ii] = 0;
		ip.pnay[ii] = 0;
		ip.pnaz[ii] = 0;
	}

	auto jsize = (nj + tile - 1) / tile;
	vector<PNAcc_Data_SoA<tile>> jpart(jsize);
	for (size_t j = 0; j < nj; ++j) {
		auto jj = j%tile;
		auto& jp = jpart[j/tile];
		jp.m[jj] = __jm[j];
		jp.e2[jj] = __je2[j];
		jp.rx[jj] = __jrdot[(0*NDIM+0)*nj + j];
		jp.ry[jj] = __jrdot[(0*NDIM+1)*nj + j];
		jp.rz[jj] = __jrdot[(0*NDIM+2)*nj + j];
		jp.vx[jj] = __jrdot[(1*NDIM+0)*nj + j];
		jp.vy[jj] = __jrdot[(1*NDIM+1)*nj + j];
		jp.vz[jj] = __jrdot[(1*NDIM+2)*nj + j];
		jp.pnax[jj] = 0;
		jp.pnay[jj] = 0;
		jp.pnaz[jj] = 0;
	}

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		P2P_pnacc_kernel_core<tile>(clight)
	);

	for (size_t i = 0; i < ni; ++i) {
		auto ii = i%tile;
		auto& ip = ipart[i/tile];
		__ipnacc[(0*NDIM+0)*ni + i] = ip.pnax[ii];
		__ipnacc[(0*NDIM+1)*ni + i] = ip.pnay[ii];
		__ipnacc[(0*NDIM+2)*ni + i] = ip.pnaz[ii];
	}

	for (size_t j = 0; j < nj; ++j) {
		auto jj = j%tile;
		auto& jp = jpart[j/tile];
		__jpnacc[(0*NDIM+0)*nj + j] = jp.pnax[jj];
		__jpnacc[(0*NDIM+1)*nj + j] = jp.pnay[jj];
		__jpnacc[(0*NDIM+2)*nj + j] = jp.pnaz[jj];
	}
}


void
pnacc_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const CLIGHT clight,
	real_t __ipnacc[])
{
	constexpr auto tile = 1;

	auto isize = (ni + tile - 1) / tile;
	vector<PNAcc_Data_SoA<tile>> ipart(isize);
	for (size_t i = 0; i < ni; ++i) {
		auto ii = i%tile;
		auto& ip = ipart[i/tile];
		ip.m[ii] = __im[i];
		ip.e2[ii] = __ie2[i];
		ip.rx[ii] = __irdot[(0*NDIM+0)*ni + i];
		ip.ry[ii] = __irdot[(0*NDIM+1)*ni + i];
		ip.rz[ii] = __irdot[(0*NDIM+2)*ni + i];
		ip.vx[ii] = __irdot[(1*NDIM+0)*ni + i];
		ip.vy[ii] = __irdot[(1*NDIM+1)*ni + i];
		ip.vz[ii] = __irdot[(1*NDIM+2)*ni + i];
		ip.pnax[ii] = 0;
		ip.pnay[ii] = 0;
		ip.pnaz[ii] = 0;
	}

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		P2P_pnacc_kernel_core<tile>(clight)
	);

	for (size_t i = 0; i < ni; ++i) {
		auto ii = i%tile;
		auto& ip = ipart[i/tile];
		__ipnacc[(0*NDIM+0)*ni + i] = ip.pnax[ii];
		__ipnacc[(0*NDIM+1)*ni + i] = ip.pnay[ii];
		__ipnacc[(0*NDIM+2)*ni + i] = ip.pnaz[ii];
	}
}


void
pnacc_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	const CLIGHT clight,
	real_t __ipnacc[])
{
	for (uint_t i = 0; i < ni; ++i) {
		PNAcc_Data ip;
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (uint_t kdot = 0; kdot < 2; ++kdot) {
			for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
			}
		}
		for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
			ip.pnacc[kdim] = 0;
		}

		for (uint_t j = 0; j < nj; ++j) {
			PNAcc_Data jp;
			jp.m = __jm[j];
			jp.e2 = __je2[j];
			for (uint_t kdot = 0; kdot < 2; ++kdot) {
				for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
					const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
					jp.rdot[kdot][kdim] = ptr[j];
				}
			}
			for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
				jp.pnacc[kdim] = 0;
			}
			ip = pnacc_kernel_core(ip, jp, clight);
		}

		for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
			real_t *ptr = &__ipnacc[kdim*ni];
			ptr[i] = ip.pnacc[kdim];
		}
	}
}

