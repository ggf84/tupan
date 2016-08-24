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
	vector<PNAcc_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
			}
		}
		for (auto kdim = 0; kdim < NDIM; ++kdim) {
			ip.pnacc[kdim] = 0;
		}
	}

	vector<PNAcc_Data> jpart{nj};
	for (auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		jp.m = __jm[j];
		jp.e2 = __je2[j];
		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
				jp.rdot[kdot][kdim] = ptr[j];
			}
		}
		for (auto kdim = 0; kdim < NDIM; ++kdim) {
			jp.pnacc[kdim] = 0;
		}
	}

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		[clight=clight](auto &ip, auto &jp)
		{
			p2p_pnacc_kernel_core(ip, jp, clight);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		for (auto kdim = 0; kdim < NDIM; ++kdim) {
			real_t *ptr = &__ipnacc[kdim*ni];
			ptr[i] = ip.pnacc[kdim];
		}
	}

	for (const auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		for (auto kdim = 0; kdim < NDIM; ++kdim) {
			real_t *ptr = &__jpnacc[kdim*nj];
			ptr[j] = jp.pnacc[kdim];
		}
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
	vector<PNAcc_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.m = __im[i];
		ip.e2 = __ie2[i];
		for (auto kdot = 0; kdot < 2; ++kdot) {
			for (auto kdim = 0; kdim < NDIM; ++kdim) {
				const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = ptr[i];
			}
		}
		for (auto kdim = 0; kdim < NDIM; ++kdim) {
			ip.pnacc[kdim] = 0;
		}
	}

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		[clight=clight](auto &ip, auto &jp)
		{
			p2p_pnacc_kernel_core(ip, jp, clight);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		for (auto kdim = 0; kdim < NDIM; ++kdim) {
			real_t *ptr = &__ipnacc[kdim*ni];
			ptr[i] = ip.pnacc[kdim];
		}
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

