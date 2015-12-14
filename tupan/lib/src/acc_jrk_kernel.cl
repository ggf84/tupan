#include "acc_jrk_kernel_common.h"


kernel void
acc_jrk_kernel(
	const uint_t ni,
	global const real_tn __im[],
	global const real_tn __irx[],
	global const real_tn __iry[],
	global const real_tn __irz[],
	global const real_tn __ie2[],
	global const real_tn __ivx[],
	global const real_tn __ivy[],
	global const real_tn __ivz[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __jrx[],
	global const real_t __jry[],
	global const real_t __jrz[],
	global const real_t __je2[],
	global const real_t __jvx[],
	global const real_t __jvy[],
	global const real_t __jvz[],
	global real_tn __iax[],
	global real_tn __iay[],
	global real_tn __iaz[],
	global real_tn __ijx[],
	global real_tn __ijy[],
	global real_tn __ijz[])
{
	uint_t lid = get_local_id(0);
	uint_t start = get_group_id(0) * get_local_size(0);
	uint_t stride = get_num_groups(0) * get_local_size(0);
	for (uint_t ii = start; ii * SIMD < ni; ii += stride) {
		uint_t i = ii + lid;
		i *= (i * SIMD < ni);

		vec(Acc_Jrk_Data) ip = (vec(Acc_Jrk_Data)){
			.ax = (real_tn)(0),
			.ay = (real_tn)(0),
			.az = (real_tn)(0),
			.jx = (real_tn)(0),
			.jy = (real_tn)(0),
			.jz = (real_tn)(0),
			.rx = __irx[i],
			.ry = __iry[i],
			.rz = __irz[i],
			.vx = __ivx[i],
			.vy = __ivy[i],
			.vz = __ivz[i],
			.e2 = __ie2[i],
			.m = __im[i],
		};

		uint_t j = 0;

		#ifdef FAST_LOCAL_MEM
		for (; ((j + LSIZE) - 1) < nj; j += LSIZE) {
			Acc_Jrk_Data jp = (Acc_Jrk_Data){
				.rx = __jrx[j + lid],
				.ry = __jry[j + lid],
				.rz = __jrz[j + lid],
				.vx = __jvx[j + lid],
				.vy = __jvy[j + lid],
				.vz = __jvz[j + lid],
				.e2 = __je2[j + lid],
				.m = __jm[j + lid],
			};
			barrier(CLK_LOCAL_MEM_FENCE);
			local Acc_Jrk_Data _jp[LSIZE];
			_jp[lid] = jp;
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll
			for (uint_t k = 0; k < LSIZE; ++k) {
				ip = acc_jrk_kernel_core(ip, _jp[k]);
			}
		}
		#endif

		for (; ((j + 1) - 1) < nj; j += 1) {
			Acc_Jrk_Data jp = (Acc_Jrk_Data){
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

