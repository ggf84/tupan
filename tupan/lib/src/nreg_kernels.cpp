#include "nbody_parallel.h"
#include "nreg_kernels_common.h"


void
nreg_Xkernel_rectangle(
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
	real_t __idrx[],
	real_t __idry[],
	real_t __idrz[],
	real_t __iax[],
	real_t __iay[],
	real_t __iaz[],
	real_t __iu[],
	real_t __jdrx[],
	real_t __jdry[],
	real_t __jdrz[],
	real_t __jax[],
	real_t __jay[],
	real_t __jaz[],
	real_t __ju[])
{
	vector<Nreg_X_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.drx = 0;
		ip.dry = 0;
		ip.drz = 0;
		ip.ax = 0;
		ip.ay = 0;
		ip.az = 0;
		ip.u = 0;
		ip.rx = __irx[i];
		ip.ry = __iry[i];
		ip.rz = __irz[i];
		ip.vx = __ivx[i];
		ip.vy = __ivy[i];
		ip.vz = __ivz[i];
		ip.e2 = __ie2[i];
		ip.m = __im[i];
	}

	vector<Nreg_X_Data> jpart{nj};
	for (auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		jp.drx = 0;
		jp.dry = 0;
		jp.drz = 0;
		jp.ax = 0;
		jp.ay = 0;
		jp.az = 0;
		jp.u = 0;
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
		[dt=dt](auto &ip, auto &jp)
		{
			p2p_nreg_Xkernel_core(ip, jp, dt);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__idrx[i] = ip.drx;
		__idry[i] = ip.dry;
		__idrz[i] = ip.drz;
		__iax[i] = ip.ax;
		__iay[i] = ip.ay;
		__iaz[i] = ip.az;
		__iu[i] = ip.m * ip.u;
	}

	for (const auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		__jdrx[j] = jp.drx;
		__jdry[j] = jp.dry;
		__jdrz[j] = jp.drz;
		__jax[j] = jp.ax;
		__jay[j] = jp.ay;
		__jaz[j] = jp.az;
		__ju[j] = jp.m * jp.u;
	}
}


void
nreg_Xkernel_triangle(
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
	real_t __idrx[],
	real_t __idry[],
	real_t __idrz[],
	real_t __iax[],
	real_t __iay[],
	real_t __iaz[],
	real_t __iu[])
{
	vector<Nreg_X_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.drx = 0;
		ip.dry = 0;
		ip.drz = 0;
		ip.ax = 0;
		ip.ay = 0;
		ip.az = 0;
		ip.u = 0;
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
		[dt=dt](auto &ip, auto &jp)
		{
			p2p_nreg_Xkernel_core(ip, jp, dt);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__idrx[i] = ip.drx;
		__idry[i] = ip.dry;
		__idrz[i] = ip.drz;
		__iax[i] = ip.ax;
		__iay[i] = ip.ay;
		__iaz[i] = ip.az;
		__iu[i] = ip.m * ip.u;
	}
}


void
nreg_Xkernel(
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
	real_t __idrx[],
	real_t __idry[],
	real_t __idrz[],
	real_t __iax[],
	real_t __iay[],
	real_t __iaz[],
	real_t __iu[])
{
	for (uint_t i = 0; i < ni; ++i) {
		Nreg_X_Data ip = (Nreg_X_Data){
			.drx = 0,
			.dry = 0,
			.drz = 0,
			.ax = 0,
			.ay = 0,
			.az = 0,
			.u = 0,
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
			Nreg_X_Data jp = (Nreg_X_Data){
				.drx = 0,
				.dry = 0,
				.drz = 0,
				.ax = 0,
				.ay = 0,
				.az = 0,
				.u = 0,
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.e2 = __je2[j],
				.m = __jm[j],
			};
			ip = nreg_Xkernel_core(ip, jp, dt);
		}

		__idrx[i] = ip.drx;
		__idry[i] = ip.dry;
		__idrz[i] = ip.drz;
		__iax[i] = ip.ax;
		__iay[i] = ip.ay;
		__iaz[i] = ip.az;
		__iu[i] = ip.m * ip.u;
	}
}


void
nreg_Vkernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const real_t __iax[],
	const real_t __iay[],
	const real_t __iaz[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __jvx[],
	const real_t __jvy[],
	const real_t __jvz[],
	const real_t __jax[],
	const real_t __jay[],
	const real_t __jaz[],
	const real_t dt,
	real_t __idvx[],
	real_t __idvy[],
	real_t __idvz[],
	real_t __ik[],
	real_t __jdvx[],
	real_t __jdvy[],
	real_t __jdvz[],
	real_t __jk[])
{
	vector<Nreg_V_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.dvx = 0;
		ip.dvy = 0;
		ip.dvz = 0;
		ip.k = 0;
		ip.vx = __ivx[i];
		ip.vy = __ivy[i];
		ip.vz = __ivz[i];
		ip.ax = __iax[i];
		ip.ay = __iay[i];
		ip.az = __iaz[i];
		ip.m = __im[i];
	}

	vector<Nreg_V_Data> jpart{nj};
	for (auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		jp.dvx = 0;
		jp.dvy = 0;
		jp.dvz = 0;
		jp.k = 0;
		jp.vx = __jvx[j];
		jp.vy = __jvy[j];
		jp.vz = __jvz[j];
		jp.ax = __jax[j];
		jp.ay = __jay[j];
		jp.az = __jaz[j];
		jp.m = __jm[j];
	}

	#pragma omp parallel
	#pragma omp single
	rectangle(
		begin(ipart), end(ipart),
		begin(jpart), end(jpart),
		[dt=dt](auto &ip, auto &jp)
		{
			p2p_nreg_Vkernel_core(ip, jp, dt);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__idvx[i] = ip.dvx;
		__idvy[i] = ip.dvy;
		__idvz[i] = ip.dvz;
		__ik[i] = ip.m * ip.k;
	}

	for (const auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		__jdvx[j] = jp.dvx;
		__jdvy[j] = jp.dvy;
		__jdvz[j] = jp.dvz;
		__jk[j] = jp.m * jp.k;
	}
}


void
nreg_Vkernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const real_t __iax[],
	const real_t __iay[],
	const real_t __iaz[],
	const real_t dt,
	real_t __idvx[],
	real_t __idvy[],
	real_t __idvz[],
	real_t __ik[])
{
	vector<Nreg_V_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.dvx = 0;
		ip.dvy = 0;
		ip.dvz = 0;
		ip.k = 0;
		ip.vx = __ivx[i];
		ip.vy = __ivy[i];
		ip.vz = __ivz[i];
		ip.ax = __iax[i];
		ip.ay = __iay[i];
		ip.az = __iaz[i];
		ip.m = __im[i];
	}

	#pragma omp parallel
	#pragma omp single
	triangle(
		begin(ipart), end(ipart),
		[dt=dt](auto &ip, auto &jp)
		{
			p2p_nreg_Vkernel_core(ip, jp, dt);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__idvx[i] = ip.dvx;
		__idvy[i] = ip.dvy;
		__idvz[i] = ip.dvz;
		__ik[i] = ip.m * ip.k;
	}
}


void
nreg_Vkernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const real_t __iax[],
	const real_t __iay[],
	const real_t __iaz[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __jvx[],
	const real_t __jvy[],
	const real_t __jvz[],
	const real_t __jax[],
	const real_t __jay[],
	const real_t __jaz[],
	const real_t dt,
	real_t __idvx[],
	real_t __idvy[],
	real_t __idvz[],
	real_t __ik[])
{
	for (uint_t i = 0; i < ni; ++i) {
		Nreg_V_Data ip = (Nreg_V_Data){
			.dvx = 0,
			.dvy = 0,
			.dvz = 0,
			.k = 0,
			.vx = __ivx[i],
			.vy = __ivy[i],
			.vz = __ivz[i],
			.ax = __iax[i],
			.ay = __iay[i],
			.az = __iaz[i],
			.m = __im[i],
		};

		for (uint_t j = 0; j < nj; ++j) {
			Nreg_V_Data jp = (Nreg_V_Data){
				.dvx = 0,
				.dvy = 0,
				.dvz = 0,
				.k = 0,
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.ax = __jax[j],
				.ay = __jay[j],
				.az = __jaz[j],
				.m = __jm[j],
			};
			ip = nreg_Vkernel_core(ip, jp, dt);
		}

		__idvx[i] = ip.dvx;
		__idvy[i] = ip.dvy;
		__idvz[i] = ip.dvz;
		__ik[i] = ip.m * ip.k;
	}
}

