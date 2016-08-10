#include "pnacc_kernel_common.h"


kernel void
pnacc_kernel(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __irx[],
	global const real_t __iry[],
	global const real_t __irz[],
	global const real_t __ie2[],
	global const real_t __ivx[],
	global const real_t __ivy[],
	global const real_t __ivz[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __jrx[],
	global const real_t __jry[],
	global const real_t __jrz[],
	global const real_t __je2[],
	global const real_t __jvx[],
	global const real_t __jvy[],
	global const real_t __jvz[],
//	const CLIGHT clight,
	constant const CLIGHT *  clight,
	global real_t __ipnax[],
	global real_t __ipnay[],
	global real_t __ipnaz[])
{
	uint_t lid = get_local_id(0);
	uint_t start = get_group_id(0) * get_local_size(0);
	uint_t stride = get_num_groups(0) * get_local_size(0);
	for (uint_t ii = start; ii * SIMD < ni; ii += stride) {
		uint_t i = ii + lid;
		i *= SIMD;
		i = (i+SIMD < ni) ? (i):(ni-SIMD);
		i *= (SIMD < ni);

		vec(PNAcc_Data) ip = (vec(PNAcc_Data)){
			.pnax = (real_tn)(0),
			.pnay = (real_tn)(0),
			.pnaz = (real_tn)(0),
			.rx = vec(vload)(0, __irx + i),
			.ry = vec(vload)(0, __iry + i),
			.rz = vec(vload)(0, __irz + i),
			.vx = vec(vload)(0, __ivx + i),
			.vy = vec(vload)(0, __ivy + i),
			.vz = vec(vload)(0, __ivz + i),
			.e2 = vec(vload)(0, __ie2 + i),
			.m = vec(vload)(0, __im + i),
		};

		uint_t j = 0;

		#ifdef FAST_LOCAL_MEM
		for (; ((j + LSIZE) - 1) < nj; j += LSIZE) {
			PNAcc_Data jp = (PNAcc_Data){
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
			local PNAcc_Data _jp[LSIZE];
			_jp[lid] = jp;
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll
			for (uint_t k = 0; k < LSIZE; ++k) {
				ip = pnacc_kernel_core(ip, _jp[k], *clight);
			}
		}
		#endif

		for (; ((j + 1) - 1) < nj; j += 1) {
			PNAcc_Data jp = (PNAcc_Data){
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.e2 = __je2[j],
				.m = __jm[j],
			};
			ip = pnacc_kernel_core(ip, jp, *clight);
		}

		vec(vstore)(ip.pnax, 0, __ipnax + i);
		vec(vstore)(ip.pnay, 0, __ipnay + i);
		vec(vstore)(ip.pnaz, 0, __ipnaz + i);
	}
}

