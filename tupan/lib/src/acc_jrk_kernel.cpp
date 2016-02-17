#include "nbody_parallel.h"
#include "acc_jrk_kernel_common.h"


void
acc_jrk_kernel_rectangle(
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
	real_t __iax[],
	real_t __iay[],
	real_t __iaz[],
	real_t __ijx[],
	real_t __ijy[],
	real_t __ijz[],
	real_t __jax[],
	real_t __jay[],
	real_t __jaz[],
	real_t __jjx[],
	real_t __jjy[],
	real_t __jjz[])
{
	vector<Acc_Jrk_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.ax = 0;
		ip.ay = 0;
		ip.az = 0;
		ip.jx = 0;
		ip.jy = 0;
		ip.jz = 0;
		ip.rx = __irx[i];
		ip.ry = __iry[i];
		ip.rz = __irz[i];
		ip.vx = __ivx[i];
		ip.vy = __ivy[i];
		ip.vz = __ivz[i];
		ip.e2 = __ie2[i];
		ip.m = __im[i];
	}

	vector<Acc_Jrk_Data> jpart{nj};
	for (auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		jp.ax = 0;
		jp.ay = 0;
		jp.az = 0;
		jp.jx = 0;
		jp.jy = 0;
		jp.jz = 0;
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
		[](auto &ip, auto &jp)
		{
			p2p_acc_jrk_kernel_core(ip, jp);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__iax[i] = ip.ax;
		__iay[i] = ip.ay;
		__iaz[i] = ip.az;
		__ijx[i] = ip.jx;
		__ijy[i] = ip.jy;
		__ijz[i] = ip.jz;
	}

	for (const auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		__jax[j] = jp.ax;
		__jay[j] = jp.ay;
		__jaz[j] = jp.az;
		__jjx[j] = jp.jx;
		__jjy[j] = jp.jy;
		__jjz[j] = jp.jz;
	}
}


void
acc_jrk_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	real_t __iax[],
	real_t __iay[],
	real_t __iaz[],
	real_t __ijx[],
	real_t __ijy[],
	real_t __ijz[])
{
	vector<Acc_Jrk_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.ax = 0;
		ip.ay = 0;
		ip.az = 0;
		ip.jx = 0;
		ip.jy = 0;
		ip.jz = 0;
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
		[](auto &ip, auto &jp)
		{
			p2p_acc_jrk_kernel_core(ip, jp);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__iax[i] = ip.ax;
		__iay[i] = ip.ay;
		__iaz[i] = ip.az;
		__ijx[i] = ip.jx;
		__ijy[i] = ip.jy;
		__ijz[i] = ip.jz;
	}
}


void
acc_jrk_kernel(
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
	real_t __iax[],
	real_t __iay[],
	real_t __iaz[],
	real_t __ijx[],
	real_t __ijy[],
	real_t __ijz[])
{
	for (uint_t i = 0; i < ni; ++i) {
		Acc_Jrk_Data ip = (Acc_Jrk_Data){
			.ax = 0,
			.ay = 0,
			.az = 0,
			.jx = 0,
			.jy = 0,
			.jz = 0,
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
			Acc_Jrk_Data jp = (Acc_Jrk_Data){
				.ax = 0,
				.ay = 0,
				.az = 0,
				.jx = 0,
				.jy = 0,
				.jz = 0,
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.e2 = __je2[j],
				.m = __jm[j],
			};
			ip = acc_jrk_kernel_core(ip, jp);
		}

		__iax[i] = ip.ax;
		__iay[i] = ip.ay;
		__iaz[i] = ip.az;
		__ijx[i] = ip.jx;
		__ijy[i] = ip.jy;
		__ijz[i] = ip.jz;
	}
}

