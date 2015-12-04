#include "nbody_parallel.h"
#include "acc_kernel_common.h"


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
	vector<Acc_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.ax = 0;
		ip.ay = 0;
		ip.az = 0;
		ip.rx = __irx[i];
		ip.ry = __iry[i];
		ip.rz = __irz[i];
		ip.e2 = __ie2[i];
		ip.m = __im[i];
	}

	vector<Acc_Data> jpart{nj};
	for (auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		jp.ax = 0;
		jp.ay = 0;
		jp.az = 0;
		jp.rx = __jrx[j];
		jp.ry = __jry[j];
		jp.rz = __jrz[j];
		jp.e2 = __je2[j];
		jp.m = __jm[j];
	}

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		p2p_acc_kernel_core{}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__iax[i] = ip.ax;
		__iay[i] = ip.ay;
		__iaz[i] = ip.az;
	}

	for (const auto& jp : jpart) {
		auto j = &jp - &jpart[0];
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
	vector<Acc_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.ax = 0;
		ip.ay = 0;
		ip.az = 0;
		ip.rx = __irx[i];
		ip.ry = __iry[i];
		ip.rz = __irz[i];
		ip.e2 = __ie2[i];
		ip.m = __im[i];
	}

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		p2p_acc_kernel_core{}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__iax[i] = ip.ax;
		__iay[i] = ip.ay;
		__iaz[i] = ip.az;
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

