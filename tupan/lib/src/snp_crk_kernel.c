#include "snp_crk_kernel_common.h"


void
snp_crk_kernel(
	const uint_t ni,
	const real_t __im[restrict],
	const real_t __irx[restrict],
	const real_t __iry[restrict],
	const real_t __irz[restrict],
	const real_t __ie2[restrict],
	const real_t __ivx[restrict],
	const real_t __ivy[restrict],
	const real_t __ivz[restrict],
	const real_t __iax[restrict],
	const real_t __iay[restrict],
	const real_t __iaz[restrict],
	const real_t __ijx[restrict],
	const real_t __ijy[restrict],
	const real_t __ijz[restrict],
	const uint_t nj,
	const real_t __jm[restrict],
	const real_t __jrx[restrict],
	const real_t __jry[restrict],
	const real_t __jrz[restrict],
	const real_t __je2[restrict],
	const real_t __jvx[restrict],
	const real_t __jvy[restrict],
	const real_t __jvz[restrict],
	const real_t __jax[restrict],
	const real_t __jay[restrict],
	const real_t __jaz[restrict],
	const real_t __jjx[restrict],
	const real_t __jjy[restrict],
	const real_t __jjz[restrict],
	real_t __isx[restrict],
	real_t __isy[restrict],
	real_t __isz[restrict],
	real_t __icx[restrict],
	real_t __icy[restrict],
	real_t __icz[restrict])
{
	for (uint_t i = 0; i < ni; ++i) {
		Snp_Crk_IData ip = (Snp_Crk_IData){
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
			Snp_Crk_JData jp = (Snp_Crk_JData){
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

