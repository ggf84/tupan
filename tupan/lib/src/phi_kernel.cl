#include "phi_kernel_common.h"


kernel void
phi_kernel(
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
	global real_tn __iphi[])
{
	uint_t lid = get_local_id(0);
	uint_t gid = get_global_id(0);
	uint_t i = gid % ni;

	Phi_IData ip = (Phi_IData){
		.phi = 0,
		.rx = __irx[i],
		.ry = __iry[i],
		.rz = __irz[i],
		.e2 = __ie2[i],
		.m = __im[i],
	};

	uint_t j = 0;

	#ifdef FAST_LOCAL_MEM
	local Phi_JData _jp[LSIZE];
	#pragma unroll
	for (; (j + LSIZE - 1) < nj; j += LSIZE) {
		barrier(CLK_LOCAL_MEM_FENCE);
		_jp[lid] = (Phi_JData){
			.rx = __jrx[j + lid],
			.ry = __jry[j + lid],
			.rz = __jrz[j + lid],
			.e2 = __je2[j + lid],
			.m = __jm[j + lid],
		};
		barrier(CLK_LOCAL_MEM_FENCE);
		#pragma unroll
		for (uint_t k = 0; k < LSIZE; ++k) {
			ip = phi_kernel_core(ip, _jp[k]);
		}
	}
	#endif

	#pragma unroll
	for (uint_t k = j; k < nj; ++k) {
		Phi_JData jp = (Phi_JData){
			.rx = __jrx[k],
			.ry = __jry[k],
			.rz = __jrz[k],
			.e2 = __je2[k],
			.m = __jm[k],
		};
		ip = phi_kernel_core(ip, jp);
	}

	__iphi[i] = ip.phi;
}

