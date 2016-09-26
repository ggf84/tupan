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
	constexpr auto tile = 4;

	auto isize = (ni + tile - 1) / tile;
	vector<Sakura_Data_SoA<tile>> ipart(isize);
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
		ip.drx[ii] = 0;
		ip.dry[ii] = 0;
		ip.drz[ii] = 0;
		ip.dvx[ii] = 0;
		ip.dvy[ii] = 0;
		ip.dvz[ii] = 0;
	}

	auto jsize = (nj + tile - 1) / tile;
	vector<Sakura_Data_SoA<tile>> jpart(jsize);
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
		jp.drx[jj] = 0;
		jp.dry[jj] = 0;
		jp.drz[jj] = 0;
		jp.dvx[jj] = 0;
		jp.dvy[jj] = 0;
		jp.dvz[jj] = 0;
	}

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		P2P_sakura_kernel_core<tile>(dt, flag)
	);

	for (size_t i = 0; i < ni; ++i) {
		auto ii = i%tile;
		auto& ip = ipart[i/tile];
		__idrdot[(0*NDIM+0)*ni + i] = ip.drx[ii];
		__idrdot[(0*NDIM+1)*ni + i] = ip.dry[ii];
		__idrdot[(0*NDIM+2)*ni + i] = ip.drz[ii];
		__idrdot[(1*NDIM+0)*ni + i] = ip.dvx[ii];
		__idrdot[(1*NDIM+1)*ni + i] = ip.dvy[ii];
		__idrdot[(1*NDIM+2)*ni + i] = ip.dvz[ii];
	}

	for (size_t j = 0; j < nj; ++j) {
		auto jj = j%tile;
		auto& jp = jpart[j/tile];
		__jdrdot[(0*NDIM+0)*nj + j] = jp.drx[jj];
		__jdrdot[(0*NDIM+1)*nj + j] = jp.dry[jj];
		__jdrdot[(0*NDIM+2)*nj + j] = jp.drz[jj];
		__jdrdot[(1*NDIM+0)*nj + j] = jp.dvx[jj];
		__jdrdot[(1*NDIM+1)*nj + j] = jp.dvy[jj];
		__jdrdot[(1*NDIM+2)*nj + j] = jp.dvz[jj];
	}
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
	constexpr auto tile = 4;

	auto isize = (ni + tile - 1) / tile;
	vector<Sakura_Data_SoA<tile>> ipart(isize);
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
		ip.drx[ii] = 0;
		ip.dry[ii] = 0;
		ip.drz[ii] = 0;
		ip.dvx[ii] = 0;
		ip.dvy[ii] = 0;
		ip.dvz[ii] = 0;
	}

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		P2P_sakura_kernel_core<tile>(dt, flag)
	);

	for (size_t i = 0; i < ni; ++i) {
		auto ii = i%tile;
		auto& ip = ipart[i/tile];
		__idrdot[(0*NDIM+0)*ni + i] = ip.drx[ii];
		__idrdot[(0*NDIM+1)*ni + i] = ip.dry[ii];
		__idrdot[(0*NDIM+2)*ni + i] = ip.drz[ii];
		__idrdot[(1*NDIM+0)*ni + i] = ip.dvx[ii];
		__idrdot[(1*NDIM+1)*ni + i] = ip.dvy[ii];
		__idrdot[(1*NDIM+2)*ni + i] = ip.dvz[ii];
	}
}


void
sakura_kernel(
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
	real_t __idrdot[])
{
	for (uint_t i = 0; i < ni; ++i) {
		Sakura_Data ip;
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
				ip.drdot[kdot][kdim] = 0;
			}
		}

		for (uint_t j = 0; j < nj; ++j) {
			Sakura_Data jp;
			jp.m = __jm[j];
			jp.e2 = __je2[j];
			for (auto kdot = 0; kdot < 2; ++kdot) {
				for (auto kdim = 0; kdim < NDIM; ++kdim) {
					const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
					jp.rdot[kdot][kdim] = ptr[j];
					jp.drdot[kdot][kdim] = 0;
				}
			}
			ip = sakura_kernel_core(ip, jp, dt, flag);
		}

		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				real_t *ptr = &__idrdot[(kdot*NDIM+kdim)*ni];
				ptr[i] = ip.drdot[kdot][kdim];
			}
		}
	}
}

