#include "snp_crk_kernel_common.h"


kernel void
snp_crk_kernel(
	const uint_t ni,
	global const real_tn __im[],
	global const real_tn __irx[],
	global const real_tn __iry[],
	global const real_tn __irz[],
	global const real_tn __ie2[],
	global const real_tn __ivx[],
	global const real_tn __ivy[],
	global const real_tn __ivz[],
	global const real_tn __iax[],
	global const real_tn __iay[],
	global const real_tn __iaz[],
	global const real_tn __ijx[],
	global const real_tn __ijy[],
	global const real_tn __ijz[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __jrx[],
	global const real_t __jry[],
	global const real_t __jrz[],
	global const real_t __je2[],
	global const real_t __jvx[],
	global const real_t __jvy[],
	global const real_t __jvz[],
	global const real_t __jax[],
	global const real_t __jay[],
	global const real_t __jaz[],
	global const real_t __jjx[],
	global const real_t __jjy[],
	global const real_t __jjz[],
	global real_tn __isx[],
	global real_tn __isy[],
	global real_tn __isz[],
	global real_tn __icx[],
	global real_tn __icy[],
	global real_tn __icz[])
{
	uint_t lid = get_local_id(0);
	uint_t start = get_group_id(0) * get_local_size(0);
	uint_t stride = get_num_groups(0) * get_local_size(0);
	for (uint_t ii = start; ii * SIMD < ni; ii += stride) {
		uint_t i = ii + lid;
		i *= (i * SIMD < ni);

		vec(Snp_Crk_Data) ip = (vec(Snp_Crk_Data)){
			.sx = (real_tn)(0),
			.sy = (real_tn)(0),
			.sz = (real_tn)(0),
			.cx = (real_tn)(0),
			.cy = (real_tn)(0),
			.cz = (real_tn)(0),
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
		for (; ((j + LSIZE) - 1) < nj; j += LSIZE) {
			Snp_Crk_Data jp = (Snp_Crk_Data){
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
			local Snp_Crk_Data _jp[LSIZE];
			_jp[lid] = jp;
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll
			for (uint_t k = 0; k < LSIZE; ++k) {
				ip = snp_crk_kernel_core(ip, _jp[k]);
			}
		}
		#endif

		for (; ((j + 1) - 1) < nj; j += 1) {
			Snp_Crk_Data jp = (Snp_Crk_Data){
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

