#include "nbody_parallel.h"
#include "acc_kernel_common.h"


void
acc_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __iadot[],
	real_t __jadot[])
{
	constexpr auto tile = 16;

	auto isize = (ni + tile - 1) / tile;
	vector<Acc_Data_SoA<tile>> ipart(isize);
	for (size_t i = 0; i < ni; ++i) {
		auto ii = i%tile;
		auto& ip = ipart[i/tile];
		ip.m[ii] = __im[i];
		ip.e2[ii] = __ie2[i];
		ip.rx[ii] = __irdot[(0*NDIM+0)*ni + i];
		ip.ry[ii] = __irdot[(0*NDIM+1)*ni + i];
		ip.rz[ii] = __irdot[(0*NDIM+2)*ni + i];
		ip.ax[ii] = 0;
		ip.ay[ii] = 0;
		ip.az[ii] = 0;
	}

	auto jsize = (nj + tile - 1) / tile;
	vector<Acc_Data_SoA<tile>> jpart(jsize);
	for (size_t j = 0; j < nj; ++j) {
		auto jj = j%tile;
		auto& jp = jpart[j/tile];
		jp.m[jj] = __jm[j];
		jp.e2[jj] = __je2[j];
		jp.rx[jj] = __jrdot[(0*NDIM+0)*nj + j];
		jp.ry[jj] = __jrdot[(0*NDIM+1)*nj + j];
		jp.rz[jj] = __jrdot[(0*NDIM+2)*nj + j];
		jp.ax[jj] = 0;
		jp.ay[jj] = 0;
		jp.az[jj] = 0;
	}

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		P2P_acc_kernel_core<tile>()
	);

	for (size_t i = 0; i < ni; ++i) {
		auto ii = i%tile;
		auto& ip = ipart[i/tile];
		__iadot[(0*NDIM+0)*ni + i] = ip.ax[ii];
		__iadot[(0*NDIM+1)*ni + i] = ip.ay[ii];
		__iadot[(0*NDIM+2)*ni + i] = ip.az[ii];
	}

	for (size_t j = 0; j < nj; ++j) {
		auto jj = j%tile;
		auto& jp = jpart[j/tile];
		__jadot[(0*NDIM+0)*nj + j] = jp.ax[jj];
		__jadot[(0*NDIM+1)*nj + j] = jp.ay[jj];
		__jadot[(0*NDIM+2)*nj + j] = jp.az[jj];
	}
}


void
acc_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iadot[])
{
	constexpr auto tile = 16;

	auto isize = (ni + tile - 1) / tile;
	vector<Acc_Data_SoA<tile>> ipart(isize);
	for (size_t i = 0; i < ni; ++i) {
		auto ii = i%tile;
		auto& ip = ipart[i/tile];
		ip.m[ii] = __im[i];
		ip.e2[ii] = __ie2[i];
		ip.rx[ii] = __irdot[(0*NDIM+0)*ni + i];
		ip.ry[ii] = __irdot[(0*NDIM+1)*ni + i];
		ip.rz[ii] = __irdot[(0*NDIM+2)*ni + i];
		ip.ax[ii] = 0;
		ip.ay[ii] = 0;
		ip.az[ii] = 0;
	}

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		P2P_acc_kernel_core<tile>()
	);

	for (size_t i = 0; i < ni; ++i) {
		auto ii = i%tile;
		auto& ip = ipart[i/tile];
		__iadot[(0*NDIM+0)*ni + i] = ip.ax[ii];
		__iadot[(0*NDIM+1)*ni + i] = ip.ay[ii];
		__iadot[(0*NDIM+2)*ni + i] = ip.az[ii];
	}
}


void
acc_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __iadot[])
{
	for (uint_t i = 0; i < ni; ++i) {
		Acc_Data ip;
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (uint_t kdot = 0; kdot < 1; ++kdot) {
			for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
				ip.adot[kdot][kdim] = 0;
			}
		}

		for (uint_t j = 0; j < nj; ++j) {
			Acc_Data jp;
			jp.m = __jm[j];
			jp.e2 = __je2[j];
			for (uint_t kdot = 0; kdot < 1; ++kdot) {
				for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
					const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
					jp.rdot[kdot][kdim] = ptr[j];
					jp.adot[kdot][kdim] = 0;
				}
			}
			ip = acc_kernel_core(ip, jp);
		}

		for (uint_t kdot = 0; kdot < 1; ++kdot) {
			for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
				real_t *ptr = &__iadot[(kdot*NDIM+kdim)*ni];
				ptr[i] = ip.adot[kdot][kdim];
			}
		}
	}
}

