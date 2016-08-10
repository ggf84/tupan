#include "nbody_parallel.h"
#include "phi_kernel_common.h"


void
phi_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __iphi[],
	real_t __jphi[])
{
	vector<Phi_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (auto kdot = 0; kdot < 1; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
			}
		}
		ip.phi = 0;
	}

	vector<Phi_Data> jpart{nj};
	for (auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		jp.m = __jm[j];
		jp.e2 = __je2[j];
		for (auto kdot = 0; kdot < 1; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
				jp.rdot[kdot][kdim] = ptr[j];
			}
		}
		jp.phi = 0;
	}

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		[](auto &ip, auto &jp)
		{
			p2p_phi_kernel_core(ip, jp);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__iphi[i] = ip.phi;
	}

	for (const auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		__jphi[j] = jp.phi;
	}
}


void
phi_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iphi[])
{
	vector<Phi_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (auto kdot = 0; kdot < 1; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
			}
		}
		ip.phi = 0;
	}

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		[](auto &ip, auto &jp)
		{
			p2p_phi_kernel_core(ip, jp);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__iphi[i] = ip.phi;
	}
}


void
phi_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __iphi[])
{
	for (uint_t i = 0; i < ni; ++i) {
		Phi_Data ip;
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (auto kdot = 0; kdot < 1; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
			}
		}
		ip.phi = 0;

		for (uint_t j = 0; j < nj; ++j) {
			Phi_Data jp;
			jp.m = __jm[j];
			jp.e2 = __je2[j];
			for (auto kdot = 0; kdot < 1; ++kdot) {
				for (auto kdim = 0; kdim < NDIM; ++kdim) {
					const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
					jp.rdot[kdot][kdim] = ptr[j];
				}
			}
			jp.phi = 0;
			ip = phi_kernel_core(ip, jp);
		}

		__iphi[i] = ip.phi;
	}
}

