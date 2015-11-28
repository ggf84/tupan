#include "acc_kernel_common.h"


kernel void
acc_kernel(
	const uint_t ni,
	global const real_tn __im[],
	global const real_tn __irx[],
	global const real_tn __iry[],
	global const real_tn __irz[],
	global const real_tn __ie2[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __jrx[],
	global const real_t __jry[],
	global const real_t __jrz[],
	global const real_t __je2[],
	global real_tn __iax[],
	global real_tn __iay[],
	global real_tn __iaz[])
{
	uint_t lid = get_local_id(0);
	uint_t gid = get_global_id(0);
	uint_t i = gid % ni;

	Acc_IData ip = (Acc_IData){
		.ax = 0,
		.ay = 0,
		.az = 0,
		.rx = __irx[i],
		.ry = __iry[i],
		.rz = __irz[i],
		.e2 = __ie2[i],
		.m = __im[i],
	};

	uint_t j = 0;

	#ifdef FAST_LOCAL_MEM
	local Acc_JData _jp[LSIZE];
	#pragma unroll
	for (; (j + LSIZE - 1) < nj; j += LSIZE) {
		barrier(CLK_LOCAL_MEM_FENCE);
		_jp[lid] = (Acc_JData){
			.rx = __jrx[j + lid],
			.ry = __jry[j + lid],
			.rz = __jrz[j + lid],
			.e2 = __je2[j + lid],
			.m = __jm[j + lid],
		};
		barrier(CLK_LOCAL_MEM_FENCE);
		#pragma unroll
		for (uint_t k = 0; k < LSIZE; ++k) {
			ip = acc_kernel_core(ip, _jp[k]);
		}
	}
	#endif

	#pragma unroll
	for (uint_t k = j; k < nj; ++k) {
		Acc_JData jp = (Acc_JData){
			.rx = __jrx[k],
			.ry = __jry[k],
			.rz = __jrz[k],
			.e2 = __je2[k],
			.m = __jm[k],
		};
		ip = acc_kernel_core(ip, jp);
	}

	__iax[i] = ip.ax;
	__iay[i] = ip.ay;
	__iaz[i] = ip.az;
}

