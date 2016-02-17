#include "nbody_parallel.h"
#include "sakura_kernel_common.h"


void
sakura_kernel_rectangle(
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
	const real_t dt,
	const int_t flag,
	real_t __idrx[],
	real_t __idry[],
	real_t __idrz[],
	real_t __idvx[],
	real_t __idvy[],
	real_t __idvz[],
	real_t __jdrx[],
	real_t __jdry[],
	real_t __jdrz[],
	real_t __jdvx[],
	real_t __jdvy[],
	real_t __jdvz[])
{
	vector<Sakura_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.drx = 0;
		ip.dry = 0;
		ip.drz = 0;
		ip.dvx = 0;
		ip.dvy = 0;
		ip.dvz = 0;
		ip.rx = __irx[i];
		ip.ry = __iry[i];
		ip.rz = __irz[i];
		ip.vx = __ivx[i];
		ip.vy = __ivy[i];
		ip.vz = __ivz[i];
		ip.e2 = __ie2[i];
		ip.m = __im[i];
	}

	vector<Sakura_Data> jpart{nj};
	for (auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		jp.drx = 0;
		jp.dry = 0;
		jp.drz = 0;
		jp.dvx = 0;
		jp.dvy = 0;
		jp.dvz = 0;
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
		[dt=dt, flag=flag](auto &ip, auto &jp)
		{
			p2p_sakura_kernel_core(ip, jp, dt, flag);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__idrx[i] = ip.drx;
		__idry[i] = ip.dry;
		__idrz[i] = ip.drz;
		__idvx[i] = ip.dvx;
		__idvy[i] = ip.dvy;
		__idvz[i] = ip.dvz;
	}

	for (const auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		__jdrx[j] = jp.drx;
		__jdry[j] = jp.dry;
		__jdrz[j] = jp.drz;
		__jdvx[j] = jp.dvx;
		__jdvy[j] = jp.dvy;
		__jdvz[j] = jp.dvz;
	}
}


void
sakura_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const real_t dt,
	const int_t flag,
	real_t __idrx[],
	real_t __idry[],
	real_t __idrz[],
	real_t __idvx[],
	real_t __idvy[],
	real_t __idvz[])
{
	vector<Sakura_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.drx = 0;
		ip.dry = 0;
		ip.drz = 0;
		ip.dvx = 0;
		ip.dvy = 0;
		ip.dvz = 0;
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
		[dt=dt, flag=flag](auto &ip, auto &jp)
		{
			p2p_sakura_kernel_core(ip, jp, dt, flag);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__idrx[i] = ip.drx;
		__idry[i] = ip.dry;
		__idrz[i] = ip.drz;
		__idvx[i] = ip.dvx;
		__idvy[i] = ip.dvy;
		__idvz[i] = ip.dvz;
	}
}


void
sakura_kernel(
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
	const real_t dt,
	const int_t flag,
	real_t __idrx[],
	real_t __idry[],
	real_t __idrz[],
	real_t __idvx[],
	real_t __idvy[],
	real_t __idvz[])
{
	for (uint_t i = 0; i < ni; ++i) {
		Sakura_Data ip = (Sakura_Data){
			.drx = 0,
			.dry = 0,
			.drz = 0,
			.dvx = 0,
			.dvy = 0,
			.dvz = 0,
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
			Sakura_Data jp = (Sakura_Data){
				.drx = 0,
				.dry = 0,
				.drz = 0,
				.dvx = 0,
				.dvy = 0,
				.dvz = 0,
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.e2 = __je2[j],
				.m = __jm[j],
			};
			ip = sakura_kernel_core(ip, jp, dt, flag);
		}

		__idrx[i] = ip.drx;
		__idry[i] = ip.dry;
		__idrz[i] = ip.drz;
		__idvx[i] = ip.dvx;
		__idvy[i] = ip.dvy;
		__idvz[i] = ip.dvz;
	}
}

