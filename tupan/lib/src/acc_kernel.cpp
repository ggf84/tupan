#include "acc_kernel_common.h"


static inline void
p2p_acc_kernel_rectangle(
	const uint_t i0, const uint_t i1, Acc_Data ipart[],
	const uint_t j0, const uint_t j1, Acc_Data jpart[])
{
	for (uint_t i = i0; i < i1; ++i) {
		for (uint_t j = j0; j < j1; ++j) {
			p2p_acc_kernel_core(&ipart[i], &jpart[j]);
		}
	}
}


static inline void
p2p_acc_kernel_triangle(
	const uint_t i0, const uint_t i1, Acc_Data ipart[])
{
	for (uint_t i = i0; i < i1; ++i) {
		for (uint_t j = i+1; j < i1; ++j) {
			p2p_acc_kernel_core(&ipart[i], &ipart[j]);
		}
	}
}


static inline void
acc_rectangle(
	const uint_t i0, const uint_t i1, Acc_Data ipart[],
	const uint_t j0, const uint_t j1, Acc_Data jpart[])
{
	const uint_t di = i1 - i0;
	const uint_t dj = j1 - j0;
	constexpr uint_t threshold = 256;

	if (di > threshold && dj > threshold) {
		const uint_t im = i0 + di / 2;
		const uint_t jm = j0 + dj / 2;

		#pragma omp task
		{ acc_rectangle(i0, im, ipart, j0, jm, jpart); }
		acc_rectangle(im, i1, ipart, jm, j1, jpart);
		#pragma omp taskwait

		#pragma omp task
		{ acc_rectangle(i0, im, ipart, jm, j1, jpart); }
		acc_rectangle(im, i1, ipart, j0, jm, jpart);
		#pragma omp taskwait
	} else {
		p2p_acc_kernel_rectangle(i0, i1, ipart, j0, j1, jpart);
	}
}


static inline void
acc_triangle(
	const uint_t i0, const uint_t i1, Acc_Data ipart[])
{
	const uint_t di = i1 - i0;
	constexpr uint_t threshold = 256;

	if (di > threshold) {
		uint_t im = i0 + di / 2;

		#pragma omp task
		{ acc_triangle(i0, im, ipart); }
		acc_triangle(im, i1, ipart);
		#pragma omp taskwait

		acc_rectangle(i0, im, ipart, im, i1, ipart);
	} else {
		p2p_acc_kernel_triangle(i0, i1, ipart);
	}
}


void
acc_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __jrx[],
	const real_t __jry[],
	const real_t __jrz[],
	const real_t __je2[],
	real_t __iax[],
	real_t __iay[],
	real_t __iaz[])
{
	for (uint_t i = 0; i < ni; ++i) {
		Acc_IData ip = (Acc_IData){
			.ax = 0,
			.ay = 0,
			.az = 0,
			.rx = __irx[i],
			.ry = __iry[i],
			.rz = __irz[i],
			.e2 = __ie2[i],
			.m = __im[i],
		};

		for (uint_t j = 0; j < nj; ++j) {
			Acc_JData jp = (Acc_JData){
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.e2 = __je2[j],
				.m = __jm[j],
			};
			ip = acc_kernel_core(ip, jp);
		}

		__iax[i] = ip.ax;
		__iay[i] = ip.ay;
		__iaz[i] = ip.az;
	}
}


void
acc_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __jrx[],
	const real_t __jry[],
	const real_t __jrz[],
	const real_t __je2[],
	real_t __iax[],
	real_t __iay[],
	real_t __iaz[],
	real_t __jax[],
	real_t __jay[],
	real_t __jaz[])
{
	Acc_Data ipart[ni];
	for (uint_t i = 0; i < ni; ++i) {
		Acc_Data ip = (Acc_Data){
			.ax = 0,
			.ay = 0,
			.az = 0,
			.rx = __irx[i],
			.ry = __iry[i],
			.rz = __irz[i],
			.e2 = __ie2[i],
			.m = __im[i],
		};
		ipart[i] = ip;
	}

	Acc_Data jpart[nj];
	for (uint_t j = 0; j < nj; ++j) {
		Acc_Data jp = (Acc_Data){
			.ax = 0,
			.ay = 0,
			.az = 0,
			.rx = __jrx[j],
			.ry = __jry[j],
			.rz = __jrz[j],
			.e2 = __je2[j],
			.m = __jm[j],
		};
		jpart[j] = jp;
	}

	#pragma omp parallel
	#pragma omp single
	acc_rectangle(0, ni, ipart, 0, nj, jpart);

	for (uint_t i = 0; i < ni; ++i) {
		Acc_Data ip = ipart[i];
		__iax[i] = ip.ax;
		__iay[i] = ip.ay;
		__iaz[i] = ip.az;
	}

	for (uint_t j = 0; j < nj; ++j) {
		Acc_Data jp = jpart[j];
		__jax[j] = jp.ax;
		__jay[j] = jp.ay;
		__jaz[j] = jp.az;
	}
}


void
acc_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	real_t __iax[],
	real_t __iay[],
	real_t __iaz[])
{
	Acc_Data ipart[ni];
	for (uint_t i = 0; i < ni; ++i) {
		Acc_Data ip = (Acc_Data){
			.ax = 0,
			.ay = 0,
			.az = 0,
			.rx = __irx[i],
			.ry = __iry[i],
			.rz = __irz[i],
			.e2 = __ie2[i],
			.m = __im[i],
		};
		ipart[i] = ip;
	}

	#pragma omp parallel
	#pragma omp single
	acc_triangle(0, ni, ipart);

	for (uint_t i = 0; i < ni; ++i) {
		Acc_Data ip = ipart[i];
		__iax[i] = ip.ax;
		__iay[i] = ip.ay;
		__iaz[i] = ip.az;
	}
}

