#include "nbody_parallel.h"
#include "snp_crk_kernel_common.h"


void
snp_crk_kernel_rectangle(
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
	vector<Snp_Crk_Data_SoA<tile>> ipart(isize);
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
		ip.ax[ii] = __irdot[(2*NDIM+0)*ni + i];
		ip.ay[ii] = __irdot[(2*NDIM+1)*ni + i];
		ip.az[ii] = __irdot[(2*NDIM+2)*ni + i];
		ip.jx[ii] = __irdot[(3*NDIM+0)*ni + i];
		ip.jy[ii] = __irdot[(3*NDIM+1)*ni + i];
		ip.jz[ii] = __irdot[(3*NDIM+2)*ni + i];
		ip.Ax[ii] = 0;
		ip.Ay[ii] = 0;
		ip.Az[ii] = 0;
		ip.Jx[ii] = 0;
		ip.Jy[ii] = 0;
		ip.Jz[ii] = 0;
		ip.Sx[ii] = 0;
		ip.Sy[ii] = 0;
		ip.Sz[ii] = 0;
		ip.Cx[ii] = 0;
		ip.Cy[ii] = 0;
		ip.Cz[ii] = 0;
	}

	auto jsize = (nj + tile - 1) / tile;
	vector<Snp_Crk_Data_SoA<tile>> jpart(jsize);
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
		jp.ax[jj] = __jrdot[(2*NDIM+0)*nj + j];
		jp.ay[jj] = __jrdot[(2*NDIM+1)*nj + j];
		jp.az[jj] = __jrdot[(2*NDIM+2)*nj + j];
		jp.jx[jj] = __jrdot[(3*NDIM+0)*nj + j];
		jp.jy[jj] = __jrdot[(3*NDIM+1)*nj + j];
		jp.jz[jj] = __jrdot[(3*NDIM+2)*nj + j];
		jp.Ax[jj] = 0;
		jp.Ay[jj] = 0;
		jp.Az[jj] = 0;
		jp.Jx[jj] = 0;
		jp.Jy[jj] = 0;
		jp.Jz[jj] = 0;
		jp.Sx[jj] = 0;
		jp.Sy[jj] = 0;
		jp.Sz[jj] = 0;
		jp.Cx[jj] = 0;
		jp.Cy[jj] = 0;
		jp.Cz[jj] = 0;
	}

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		P2P_snp_crk_kernel_core<tile>()
	);

	for (size_t i = 0; i < ni; ++i) {
		auto ii = i%tile;
		auto& ip = ipart[i/tile];
		__iadot[(0*NDIM+0)*ni + i] = ip.Ax[ii];
		__iadot[(0*NDIM+1)*ni + i] = ip.Ay[ii];
		__iadot[(0*NDIM+2)*ni + i] = ip.Az[ii];
		__iadot[(1*NDIM+0)*ni + i] = ip.Jx[ii];
		__iadot[(1*NDIM+1)*ni + i] = ip.Jy[ii];
		__iadot[(1*NDIM+2)*ni + i] = ip.Jz[ii];
		__iadot[(2*NDIM+0)*ni + i] = ip.Sx[ii];
		__iadot[(2*NDIM+1)*ni + i] = ip.Sy[ii];
		__iadot[(2*NDIM+2)*ni + i] = ip.Sz[ii];
		__iadot[(3*NDIM+0)*ni + i] = ip.Cx[ii];
		__iadot[(3*NDIM+1)*ni + i] = ip.Cy[ii];
		__iadot[(3*NDIM+2)*ni + i] = ip.Cz[ii];
	}

	for (size_t j = 0; j < nj; ++j) {
		auto jj = j%tile;
		auto& jp = jpart[j/tile];
		__jadot[(0*NDIM+0)*nj + j] = jp.Ax[jj];
		__jadot[(0*NDIM+1)*nj + j] = jp.Ay[jj];
		__jadot[(0*NDIM+2)*nj + j] = jp.Az[jj];
		__jadot[(1*NDIM+0)*nj + j] = jp.Jx[jj];
		__jadot[(1*NDIM+1)*nj + j] = jp.Jy[jj];
		__jadot[(1*NDIM+2)*nj + j] = jp.Jz[jj];
		__jadot[(2*NDIM+0)*nj + j] = jp.Sx[jj];
		__jadot[(2*NDIM+1)*nj + j] = jp.Sy[jj];
		__jadot[(2*NDIM+2)*nj + j] = jp.Sz[jj];
		__jadot[(3*NDIM+0)*nj + j] = jp.Cx[jj];
		__jadot[(3*NDIM+1)*nj + j] = jp.Cy[jj];
		__jadot[(3*NDIM+2)*nj + j] = jp.Cz[jj];
	}
}


void
snp_crk_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iadot[])
{
	constexpr auto tile = 16;

	auto isize = (ni + tile - 1) / tile;
	vector<Snp_Crk_Data_SoA<tile>> ipart(isize);
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
		ip.ax[ii] = __irdot[(2*NDIM+0)*ni + i];
		ip.ay[ii] = __irdot[(2*NDIM+1)*ni + i];
		ip.az[ii] = __irdot[(2*NDIM+2)*ni + i];
		ip.jx[ii] = __irdot[(3*NDIM+0)*ni + i];
		ip.jy[ii] = __irdot[(3*NDIM+1)*ni + i];
		ip.jz[ii] = __irdot[(3*NDIM+2)*ni + i];
		ip.Ax[ii] = 0;
		ip.Ay[ii] = 0;
		ip.Az[ii] = 0;
		ip.Jx[ii] = 0;
		ip.Jy[ii] = 0;
		ip.Jz[ii] = 0;
		ip.Sx[ii] = 0;
		ip.Sy[ii] = 0;
		ip.Sz[ii] = 0;
		ip.Cx[ii] = 0;
		ip.Cy[ii] = 0;
		ip.Cz[ii] = 0;
	}

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		P2P_snp_crk_kernel_core<tile>()
	);

	for (size_t i = 0; i < ni; ++i) {
		auto ii = i%tile;
		auto& ip = ipart[i/tile];
		__iadot[(0*NDIM+0)*ni + i] = ip.Ax[ii];
		__iadot[(0*NDIM+1)*ni + i] = ip.Ay[ii];
		__iadot[(0*NDIM+2)*ni + i] = ip.Az[ii];
		__iadot[(1*NDIM+0)*ni + i] = ip.Jx[ii];
		__iadot[(1*NDIM+1)*ni + i] = ip.Jy[ii];
		__iadot[(1*NDIM+2)*ni + i] = ip.Jz[ii];
		__iadot[(2*NDIM+0)*ni + i] = ip.Sx[ii];
		__iadot[(2*NDIM+1)*ni + i] = ip.Sy[ii];
		__iadot[(2*NDIM+2)*ni + i] = ip.Sz[ii];
		__iadot[(3*NDIM+0)*ni + i] = ip.Cx[ii];
		__iadot[(3*NDIM+1)*ni + i] = ip.Cy[ii];
		__iadot[(3*NDIM+2)*ni + i] = ip.Cz[ii];
	}
}


void
snp_crk_kernel(
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
		Snp_Crk_Data ip;
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (uint_t kdot = 0; kdot < 4; ++kdot) {
			for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
				ip.adot[kdot][kdim] = 0;
			}
		}

		for (uint_t j = 0; j < nj; ++j) {
			Snp_Crk_Data jp;
			jp.m = __jm[j];
			jp.e2 = __je2[j];
			for (uint_t kdot = 0; kdot < 4; ++kdot) {
				for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
					const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
					jp.rdot[kdot][kdim] = ptr[j];
					jp.adot[kdot][kdim] = 0;
				}
			}
			ip = snp_crk_kernel_core(ip, jp);
		}

		for (uint_t kdot = 0; kdot < 4; ++kdot) {
			for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
				real_t *ptr = &__iadot[(kdot*NDIM+kdim)*ni];
				ptr[i] = ip.adot[kdot][kdim];
			}
		}
	}
}

