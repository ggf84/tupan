#include "nbody_parallel.h"
#include "tstep_kernel_common.h"


void
tstep_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	const real_t eta,
	real_t __idt_a[],
	real_t __idt_b[],
	real_t __jdt_a[],
	real_t __jdt_b[])
{
	constexpr auto tile = 16;

	auto isize = (ni + tile - 1) / tile;
	vector<Tstep_Data_SoA<tile>> ipart(isize);
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
		ip.w2_a[ii] = 0;
		ip.w2_b[ii] = 0;
	}

	auto jsize = (nj + tile - 1) / tile;
	vector<Tstep_Data_SoA<tile>> jpart(jsize);
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
		jp.w2_a[jj] = 0;
		jp.w2_b[jj] = 0;
	}

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		P2P_tstep_kernel_core<tile>(eta)
	);

	for (size_t i = 0; i < ni; ++i) {
		auto ii = i%tile;
		auto& ip = ipart[i/tile];
		__idt_a[i] = eta / sqrt(ip.w2_a[ii]);
		__idt_b[i] = eta / sqrt(ip.w2_b[ii]);
	}

	for (size_t j = 0; j < nj; ++j) {
		auto jj = j%tile;
		auto& jp = jpart[j/tile];
		__jdt_a[j] = eta / sqrt(jp.w2_a[jj]);
		__jdt_b[j] = eta / sqrt(jp.w2_b[jj]);
	}
}


void
tstep_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const real_t eta,
	real_t __idt_a[],
	real_t __idt_b[])
{
	constexpr auto tile = 16;

	auto isize = (ni + tile - 1) / tile;
	vector<Tstep_Data_SoA<tile>> ipart(isize);
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
		ip.w2_a[ii] = 0;
		ip.w2_b[ii] = 0;
	}

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		P2P_tstep_kernel_core<tile>(eta)
	);

	for (size_t i = 0; i < ni; ++i) {
		auto ii = i%tile;
		auto& ip = ipart[i/tile];
		__idt_a[i] = eta / sqrt(ip.w2_a[ii]);
		__idt_b[i] = eta / sqrt(ip.w2_b[ii]);
	}
}


void
tstep_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	const real_t eta,
	real_t __idt_a[],
	real_t __idt_b[])
{
	for (uint_t i = 0; i < ni; ++i) {
		Tstep_Data ip;
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (uint_t kdot = 0; kdot < 2; ++kdot) {
			for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
			}
			ip.w2[kdot] = 0;
		}

		for (uint_t j = 0; j < nj; ++j) {
			Tstep_Data jp;
			jp.m = __jm[j];
			jp.e2 = __je2[j];
			for (uint_t kdot = 0; kdot < 2; ++kdot) {
				for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
					const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
					jp.rdot[kdot][kdim] = ptr[j];
				}
				jp.w2[kdot] = 0;
			}
			ip = tstep_kernel_core(ip, jp, eta);
		}

		__idt_a[i] = eta / sqrt(ip.w2[0]);
		__idt_b[i] = eta / sqrt(ip.w2[1]);
	}
}

