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
	vector<Snp_Crk_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (auto kdot = 0; kdot < 4; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
				ip.adot[kdot][kdim] = 0;
			}
		}
	}

	vector<Snp_Crk_Data> jpart{nj};
	for (auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		jp.m = __jm[j];
		jp.e2 = __je2[j];
		for (auto kdot = 0; kdot < 4; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
				jp.rdot[kdot][kdim] = ptr[j];
				jp.adot[kdot][kdim] = 0;
			}
		}
	}

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		[](auto &ip, auto &jp)
		{
			p2p_snp_crk_kernel_core(ip, jp);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		for (auto kdot = 0; kdot < 4; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				real_t *ptr = &__iadot[(kdot*NDIM+kdim)*ni];
				ptr[i] = ip.adot[kdot][kdim];
			}
		}
	}

	for (const auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		for (auto kdot = 0; kdot < 4; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				real_t *ptr = &__jadot[(kdot*NDIM+kdim)*nj];
				ptr[j] = jp.adot[kdot][kdim];
			}
		}
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
	vector<Snp_Crk_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (auto kdot = 0; kdot < 4; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
				ip.adot[kdot][kdim] = 0;
			}
		}
	}

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		[](auto &ip, auto &jp)
		{
			p2p_snp_crk_kernel_core(ip, jp);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		for (auto kdot = 0; kdot < 4; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				real_t *ptr = &__iadot[(kdot*NDIM+kdim)*ni];
				ptr[i] = ip.adot[kdot][kdim];
			}
		}
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

