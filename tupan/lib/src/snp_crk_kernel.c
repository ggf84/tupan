#include "snp_crk_kernel_common.h"


void
snp_crk_kernel(
	uint_t const ni,
	real_t const __im[restrict],
	real_t const __irx[restrict],
	real_t const __iry[restrict],
	real_t const __irz[restrict],
	real_t const __ie2[restrict],
	real_t const __ivx[restrict],
	real_t const __ivy[restrict],
	real_t const __ivz[restrict],
	real_t const __iax[restrict],
	real_t const __iay[restrict],
	real_t const __iaz[restrict],
	real_t const __ijx[restrict],
	real_t const __ijy[restrict],
	real_t const __ijz[restrict],
	uint_t const nj,
	real_t const __jm[restrict],
	real_t const __jrx[restrict],
	real_t const __jry[restrict],
	real_t const __jrz[restrict],
	real_t const __je2[restrict],
	real_t const __jvx[restrict],
	real_t const __jvy[restrict],
	real_t const __jvz[restrict],
	real_t const __jax[restrict],
	real_t const __jay[restrict],
	real_t const __jaz[restrict],
	real_t const __jjx[restrict],
	real_t const __jjy[restrict],
	real_t const __jjz[restrict],
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

