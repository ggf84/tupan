#include "snp_crk_kernel_common.h"


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

