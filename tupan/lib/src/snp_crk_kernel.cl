#include "snp_crk_kernel_common.h"


kernel void
snp_crk_kernel(
	uint_t const ni,
	global real_tn const __im[restrict],
	global real_tn const __irx[restrict],
	global real_tn const __iry[restrict],
	global real_tn const __irz[restrict],
	global real_tn const __ie2[restrict],
	global real_tn const __ivx[restrict],
	global real_tn const __ivy[restrict],
	global real_tn const __ivz[restrict],
	global real_tn const __iax[restrict],
	global real_tn const __iay[restrict],
	global real_tn const __iaz[restrict],
	global real_tn const __ijx[restrict],
	global real_tn const __ijy[restrict],
	global real_tn const __ijz[restrict],
	uint_t const nj,
	global real_t const __jm[restrict],
	global real_t const __jrx[restrict],
	global real_t const __jry[restrict],
	global real_t const __jrz[restrict],
	global real_t const __je2[restrict],
	global real_t const __jvx[restrict],
	global real_t const __jvy[restrict],
	global real_t const __jvz[restrict],
	global real_t const __jax[restrict],
	global real_t const __jay[restrict],
	global real_t const __jaz[restrict],
	global real_t const __jjx[restrict],
	global real_t const __jjy[restrict],
	global real_t const __jjz[restrict],
	global real_tn __isx[restrict],
	global real_tn __isy[restrict],
	global real_tn __isz[restrict],
	global real_tn __icx[restrict],
	global real_tn __icy[restrict],
	global real_tn __icz[restrict])
{
	uint_t lid = get_local_id(0);
	uint_t gid = get_global_id(0);
	uint_t i = gid % ni;

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

	uint_t j = 0;

	#ifdef FAST_LOCAL_MEM
	local Snp_Crk_JData _jp[LSIZE];
	#pragma unroll
	for (; (j + LSIZE - 1) < nj; j += LSIZE) {
		barrier(CLK_LOCAL_MEM_FENCE);
		_jp[lid] = (Snp_Crk_JData){
			.rx = __jrx[j + lid],
			.ry = __jry[j + lid],
			.rz = __jrz[j + lid],
			.vx = __jvx[j + lid],
			.vy = __jvy[j + lid],
			.vz = __jvz[j + lid],
			.ax = __jax[j + lid],
			.ay = __jay[j + lid],
			.az = __jaz[j + lid],
			.jx = __jjx[j + lid],
			.jy = __jjy[j + lid],
			.jz = __jjz[j + lid],
			.e2 = __je2[j + lid],
			.m = __jm[j + lid],
		};
		barrier(CLK_LOCAL_MEM_FENCE);
		#pragma unroll
		for (uint_t k = 0; k < LSIZE; ++k) {
			ip = snp_crk_kernel_core(ip, _jp[k]);
		}
	}
	#endif

	#pragma unroll
	for (uint_t k = j; k < nj; ++k) {
		Snp_Crk_JData jp = (Snp_Crk_JData){
			.rx = __jrx[k],
			.ry = __jry[k],
			.rz = __jrz[k],
			.vx = __jvx[k],
			.vy = __jvy[k],
			.vz = __jvz[k],
			.ax = __jax[k],
			.ay = __jay[k],
			.az = __jaz[k],
			.jx = __jjx[k],
			.jy = __jjy[k],
			.jz = __jjz[k],
			.e2 = __je2[k],
			.m = __jm[k],
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

