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
	vector<Sakura_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
				ip.drdot[kdot][kdim] = 0;
			}
		}
	}

	vector<Sakura_Data> jpart{nj};
	for (auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		jp.m = __jm[j];
		jp.e2 = __je2[j];
		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
				jp.rdot[kdot][kdim] = ptr[j];
				jp.drdot[kdot][kdim] = 0;
			}
		}
	}

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		[dt=dt, flag=flag](auto &ip, auto &jp)
		{
			p2p_sakura_kernel_core(ip, jp, dt, flag);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				real_t *ptr = &__idrdot[(kdot*NDIM+kdim)*ni];
				ptr[i] = ip.drdot[kdot][kdim];
			}
		}
	}

	for (const auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				real_t *ptr = &__jdrdot[(kdot*NDIM+kdim)*nj];
				ptr[j] = jp.drdot[kdot][kdim];
			}
		}
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
	vector<Sakura_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
				ip.drdot[kdot][kdim] = 0;
			}
		}
	}

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		[dt=dt, flag=flag](auto &ip, auto &jp)
		{
			p2p_sakura_kernel_core(ip, jp, dt, flag);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				real_t *ptr = &__idrdot[(kdot*NDIM+kdim)*ni];
				ptr[i] = ip.drdot[kdot][kdim];
			}
		}
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

