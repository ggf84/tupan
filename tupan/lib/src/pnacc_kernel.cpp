#include "nbody_parallel.h"
#include "pnacc_kernel_common.h"


void
pnacc_kernel_rectangle(
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
	const CLIGHT clight,
	real_t __ipnax[],
	real_t __ipnay[],
	real_t __ipnaz[],
	real_t __jpnax[],
	real_t __jpnay[],
	real_t __jpnaz[])
{
	vector<PNAcc_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.pnax = 0;
		ip.pnay = 0;
		ip.pnaz = 0;
		ip.rx = __irx[i];
		ip.ry = __iry[i];
		ip.rz = __irz[i];
		ip.vx = __ivx[i];
		ip.vy = __ivy[i];
		ip.vz = __ivz[i];
		ip.e2 = __ie2[i];
		ip.m = __im[i];
	}

	vector<PNAcc_Data> jpart{nj};
	for (auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		jp.pnax = 0;
		jp.pnay = 0;
		jp.pnaz = 0;
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
		[clight=clight](auto &ip, auto &jp)
		{
			p2p_pnacc_kernel_core(ip, jp, clight);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__ipnax[i] = ip.pnax;
		__ipnay[i] = ip.pnay;
		__ipnaz[i] = ip.pnaz;
	}

	for (const auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		__jpnax[j] = jp.pnax;
		__jpnay[j] = jp.pnay;
		__jpnaz[j] = jp.pnaz;
	}
}


void
pnacc_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const CLIGHT clight,
	real_t __ipnax[],
	real_t __ipnay[],
	real_t __ipnaz[])
{
	vector<PNAcc_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.pnax = 0;
		ip.pnay = 0;
		ip.pnaz = 0;
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
		[clight=clight](auto &ip, auto &jp)
		{
			p2p_pnacc_kernel_core(ip, jp, clight);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__ipnax[i] = ip.pnax;
		__ipnay[i] = ip.pnay;
		__ipnaz[i] = ip.pnaz;
	}
}


void
pnacc_kernel(
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
	const CLIGHT clight,
	real_t __ipnax[],
	real_t __ipnay[],
	real_t __ipnaz[])
{
	for (uint_t i = 0; i < ni; ++i) {
		PNAcc_Data ip = (PNAcc_Data){
			.pnax = 0,
			.pnay = 0,
			.pnaz = 0,
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
			PNAcc_Data jp = (PNAcc_Data){
				.pnax = 0,
				.pnay = 0,
				.pnaz = 0,
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.e2 = __je2[j],
				.m = __jm[j],
			};
			ip = pnacc_kernel_core(ip, jp, clight);
		}

		__ipnax[i] = ip.pnax;
		__ipnay[i] = ip.pnay;
		__ipnaz[i] = ip.pnaz;
	}
}

