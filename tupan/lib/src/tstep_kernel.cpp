#include "nbody_parallel.h"
#include "tstep_kernel_common.h"


void
tstep_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __jrx[],
	const real_t __jry[],
	const real_t __jrz[],
	const real_t __je2[],
	const real_t __jvx[],
	const real_t __jvy[],
	const real_t __jvz[],
	const real_t eta,
	real_t __idt_a[],
	real_t __idt_b[],
	real_t __jdt_a[],
	real_t __jdt_b[])
{
	vector<Tstep_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.w2_a = 0;
		ip.w2_b = 0;
		ip.rx = __irx[i];
		ip.ry = __iry[i];
		ip.rz = __irz[i];
		ip.vx = __ivx[i];
		ip.vy = __ivy[i];
		ip.vz = __ivz[i];
		ip.e2 = __ie2[i];
		ip.m = __im[i];
	}

	vector<Tstep_Data> jpart{nj};
	for (auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		jp.w2_a = 0;
		jp.w2_b = 0;
		jp.rx = __jrx[j];
		jp.ry = __jry[j];
		jp.rz = __jrz[j];
		jp.vx = __jvx[j];
		jp.vy = __jvy[j];
		jp.vz = __jvz[j];
		jp.e2 = __je2[j];
		jp.m = __jm[j];
	}

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		[eta=eta](auto &ip, auto &jp)
		{
			p2p_tstep_kernel_core(ip, jp, eta);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__idt_a[i] = eta / sqrt(fmax((real_t)(1), ip.w2_a));
		__idt_b[i] = eta / sqrt(fmax((real_t)(1), ip.w2_b));
	}

	for (const auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		__jdt_a[j] = eta / sqrt(fmax((real_t)(1), jp.w2_a));
		__jdt_b[j] = eta / sqrt(fmax((real_t)(1), jp.w2_b));
	}
}


void
tstep_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const real_t eta,
	real_t __idt_a[],
	real_t __idt_b[])
{
	vector<Tstep_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.w2_a = 0;
		ip.w2_b = 0;
		ip.rx = __irx[i];
		ip.ry = __iry[i];
		ip.rz = __irz[i];
		ip.vx = __ivx[i];
		ip.vy = __ivy[i];
		ip.vz = __ivz[i];
		ip.e2 = __ie2[i];
		ip.m = __im[i];
	}

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		[eta=eta](auto &ip, auto &jp)
		{
			p2p_tstep_kernel_core(ip, jp, eta);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__idt_a[i] = eta / sqrt(fmax((real_t)(1), ip.w2_a));
		__idt_b[i] = eta / sqrt(fmax((real_t)(1), ip.w2_b));
	}
}


void
tstep_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __jrx[],
	const real_t __jry[],
	const real_t __jrz[],
	const real_t __je2[],
	const real_t __jvx[],
	const real_t __jvy[],
	const real_t __jvz[],
	const real_t eta,
	real_t __idt_a[],
	real_t __idt_b[])
{
	for (uint_t i = 0; i < ni; ++i) {
		Tstep_Data ip = (Tstep_Data){
			.w2_a = 0,
			.w2_b = 0,
			.rx = __irx[i],
			.ry = __iry[i],
			.rz = __irz[i],
			.vx = __ivx[i],
			.vy = __ivy[i],
			.vz = __ivz[i],
			.e2 = __ie2[i],
			.m = __im[i],
		};

		for (uint_t j = 0; j < nj; ++j) {
			Tstep_Data jp = (Tstep_Data){
				.w2_a = 0,
				.w2_b = 0,
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.e2 = __je2[j],
				.m = __jm[j],
			};
			ip = tstep_kernel_core(ip, jp, eta);
		}

		__idt_a[i] = eta / sqrt(fmax((real_t)(1), ip.w2_a));
		__idt_b[i] = eta / sqrt(fmax((real_t)(1), ip.w2_b));
	}
}

