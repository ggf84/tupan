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
	uint_t start = get_group_id(0) * get_local_size(0);
	uint_t stride = get_num_groups(0) * get_local_size(0);
	for (uint_t ii = start; ii * SIMD < ni; ii += stride) {
		uint_t i = ii + lid;
		i *= (i * SIMD < ni);

		vec(Phi_Data) ip = (vec(Phi_Data)){
			.phi = (real_tn)(0),
			.rx = __irx[i],
			.ry = __iry[i],
			.rz = __irz[i],
			.e2 = __ie2[i],
			.m = __im[i],
		};

		uint_t j = 0;

		#ifdef FAST_LOCAL_MEM
		for (; ((j + LSIZE) - 1) < nj; j += LSIZE) {
			Phi_Data jp = (Phi_Data){
				.rx = __jrx[j + lid],
				.ry = __jry[j + lid],
				.rz = __jrz[j + lid],
				.e2 = __je2[j + lid],
				.m = __jm[j + lid],
			};
			barrier(CLK_LOCAL_MEM_FENCE);
			local Phi_Data _jp[LSIZE];
			_jp[lid] = jp;
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll
			for (uint_t k = 0; k < LSIZE; ++k) {
				ip = phi_kernel_core(ip, _jp[k]);
			}
		}
		#endif

		for (; ((j + 1) - 1) < nj; j += 1) {
			Phi_Data jp = (Phi_Data){
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.e2 = __je2[j],
				.m = __jm[j],
			};
			ip = phi_kernel_core(ip, jp);
		}

		__iphi[i] = ip.phi;
	}
}

