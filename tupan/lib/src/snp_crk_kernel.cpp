#include "nbody_parallel.h"
#include "snp_crk_kernel_common.h"


void
snp_crk_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const real_t __iax[],
	const real_t __iay[],
	const real_t __iaz[],
	const real_t __ijx[],
	const real_t __ijy[],
	const real_t __ijz[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __jrx[],
	const real_t __jry[],
	const real_t __jrz[],
	const real_t __je2[],
	const real_t __jvx[],
	const real_t __jvy[],
	const real_t __jvz[],
	const real_t __jax[],
	const real_t __jay[],
	const real_t __jaz[],
	const real_t __jjx[],
	const real_t __jjy[],
	const real_t __jjz[],
	real_t __isx[],
	real_t __isy[],
	real_t __isz[],
	real_t __icx[],
	real_t __icy[],
	real_t __icz[],
	real_t __jsx[],
	real_t __jsy[],
	real_t __jsz[],
	real_t __jcx[],
	real_t __jcy[],
	real_t __jcz[])
{
	vector<Snp_Crk_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.sx = 0;
		ip.sy = 0;
		ip.sz = 0;
		ip.cx = 0;
		ip.cy = 0;
		ip.cz = 0;
		ip.rx = __irx[i];
		ip.ry = __iry[i];
		ip.rz = __irz[i];
		ip.vx = __ivx[i];
		ip.vy = __ivy[i];
		ip.vz = __ivz[i];
		ip.ax = __iax[i];
		ip.ay = __iay[i];
		ip.az = __iaz[i];
		ip.jx = __ijx[i];
		ip.jy = __ijy[i];
		ip.jz = __ijz[i];
		ip.e2 = __ie2[i];
		ip.m = __im[i];
	}

	vector<Snp_Crk_Data> jpart{nj};
	for (auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		jp.sx = 0;
		jp.sy = 0;
		jp.sz = 0;
		jp.cx = 0;
		jp.cy = 0;
		jp.cz = 0;
		jp.rx = __jrx[j];
		jp.ry = __jry[j];
		jp.rz = __jrz[j];
		jp.vx = __jvx[j];
		jp.vy = __jvy[j];
		jp.vz = __jvz[j];
		jp.ax = __jax[j];
		jp.ay = __jay[j];
		jp.az = __jaz[j];
		jp.jx = __jjx[j];
		jp.jy = __jjy[j];
		jp.jz = __jjz[j];
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
			p2p_snp_crk_kernel_core(ip, jp);
		}
	);

	for (const auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__isx[i] = ip.sx;
		__isy[i] = ip.sy;
		__isz[i] = ip.sz;
		__icx[i] = ip.cx;
		__icy[i] = ip.cy;
		__icz[i] = ip.cz;
	}

	for (const auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		__jsx[j] = jp.sx;
		__jsy[j] = jp.sy;
		__jsz[j] = jp.sz;
		__jcx[j] = jp.cx;
		__jcy[j] = jp.cy;
		__jcz[j] = jp.cz;
	}
}


void
snp_crk_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const real_t __iax[],
	const real_t __iay[],
	const real_t __iaz[],
	const real_t __ijx[],
	const real_t __ijy[],
	const real_t __ijz[],
	real_t __isx[],
	real_t __isy[],
	real_t __isz[],
	real_t __icx[],
	real_t __icy[],
	real_t __icz[])
{
	vector<Snp_Crk_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.sx = 0;
		ip.sy = 0;
		ip.sz = 0;
		ip.cx = 0;
		ip.cy = 0;
		ip.cz = 0;
		ip.rx = __irx[i];
		ip.ry = __iry[i];
		ip.rz = __irz[i];
		ip.vx = __ivx[i];
		ip.vy = __ivy[i];
		ip.vz = __ivz[i];
		ip.ax = __iax[i];
		ip.ay = __iay[i];
		ip.az = __iaz[i];
		ip.jx = __ijx[i];
		ip.jy = __ijy[i];
		ip.jz = __ijz[i];
		ip.e2 = __ie2[i];
		ip.m = __im[i];
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
		__isx[i] = ip.sx;
		__isy[i] = ip.sy;
		__isz[i] = ip.sz;
		__icx[i] = ip.cx;
		__icy[i] = ip.cy;
		__icz[i] = ip.cz;
	}
}


void
snp_crk_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const real_t __iax[],
	const real_t __iay[],
	const real_t __iaz[],
	const real_t __ijx[],
	const real_t __ijy[],
	const real_t __ijz[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __jrx[],
	const real_t __jry[],
	const real_t __jrz[],
	const real_t __je2[],
	const real_t __jvx[],
	const real_t __jvy[],
	const real_t __jvz[],
	const real_t __jax[],
	const real_t __jay[],
	const real_t __jaz[],
	const real_t __jjx[],
	const real_t __jjy[],
	const real_t __jjz[],
	real_t __isx[],
	real_t __isy[],
	real_t __isz[],
	real_t __icx[],
	real_t __icy[],
	real_t __icz[])
{
	for (uint_t i = 0; i < ni; ++i) {
		Snp_Crk_Data ip = (Snp_Crk_Data){
			.sx = 0,
			.sy = 0,
			.sz = 0,
			.cx = 0,
			.cy = 0,
			.cz = 0,
			.rx = __irx[i],
			.ry = __iry[i],
			.rz = __irz[i],
			.vx = __ivx[i],
			.vy = __ivy[i],
			.vz = __ivz[i],
			.ax = __iax[i],
			.ay = __iay[i],
			.az = __iaz[i],
			.jx = __ijx[i],
			.jy = __ijy[i],
			.jz = __ijz[i],
			.e2 = __ie2[i],
			.m = __im[i],
		};

		for (uint_t j = 0; j < nj; ++j) {
			Snp_Crk_Data jp = (Snp_Crk_Data){
				.sx = 0,
				.sy = 0,
				.sz = 0,
				.cx = 0,
				.cy = 0,
				.cz = 0,
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.ax = __jax[j],
				.ay = __jay[j],
				.az = __jaz[j],
				.jx = __jjx[j],
				.jy = __jjy[j],
				.jz = __jjz[j],
				.e2 = __je2[j],
				.m = __jm[j],
			};
			ip = snp_crk_kernel_core(ip, jp);
		}

		__isx[i] = ip.sx;
		__isy[i] = ip.sy;
		__isz[i] = ip.sz;
		__icx[i] = ip.cx;
		__icy[i] = ip.cy;
		__icz[i] = ip.cz;
	}
}

