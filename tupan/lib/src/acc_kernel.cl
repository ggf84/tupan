#include "acc_kernel_common.h"


kernel void
acc_kernel(
	uint_t const ni,
	global real_tn const __im[restrict],
	global real_tn const __irx[restrict],
	global real_tn const __iry[restrict],
	global real_tn const __irz[restrict],
	global real_tn const __ie2[restrict],
	uint_t const nj,
	global real_t const __jm[restrict],
	global real_t const __jrx[restrict],
	global real_t const __jry[restrict],
	global real_t const __jrz[restrict],
	global real_t const __je2[restrict],
	global real_tn __iax[restrict],
	global real_tn __iay[restrict],
	global real_tn __iaz[restrict])
{
	uint_t lid = get_local_id(0);
	uint_t gid = get_global_id(0);
	uint_t i = gid % ni;

	real_tn im = __im[i];
	real_tn irx = __irx[i];
	real_tn iry = __iry[i];
	real_tn irz = __irz[i];
	real_tn ie2 = __ie2[i];

	real_tn iax = (real_tn)(0);
	real_tn iay = (real_tn)(0);
	real_tn iaz = (real_tn)(0);

	uint_t j = 0;

	#ifdef FAST_LOCAL_MEM
	local real_t _jm[LSIZE];
	local real_t _jrx[LSIZE];
	local real_t _jry[LSIZE];
	local real_t _jrz[LSIZE];
	local real_t _je2[LSIZE];
	#pragma unroll
	for (; (j + LSIZE - 1) < nj; j += LSIZE) {
		barrier(CLK_LOCAL_MEM_FENCE);
		_jm[lid] = __jm[j + lid];
		_jrx[lid] = __jrx[j + lid];
		_jry[lid] = __jry[j + lid];
		_jrz[lid] = __jrz[j + lid];
		_je2[lid] = __je2[j + lid];
		barrier(CLK_LOCAL_MEM_FENCE);
		#pragma unroll
		for (uint_t k = 0; k < LSIZE; ++k) {
			real_t jm = _jm[k];
			real_t jrx = _jrx[k];
			real_t jry = _jry[k];
			real_t jrz = _jrz[k];
			real_t je2 = _je2[k];
			acc_kernel_core(
				im, irx, iry, irz, ie2,
				jm, jrx, jry, jrz, je2,
				&iax, &iay, &iaz);
		}
	}
	#endif

	#pragma unroll
	for (uint_t k = j; k < nj; ++k) {
		real_t jm = __jm[k];
		real_t jrx = __jrx[k];
		real_t jry = __jry[k];
		real_t jrz = __jrz[k];
		real_t je2 = __je2[k];
		acc_kernel_core(
			im, irx, iry, irz, ie2,
			jm, jrx, jry, jrz, je2,
			&iax, &iay, &iaz);
	}

	__iax[i] = iax;
	__iay[i] = iay;
	__iaz[i] = iaz;
}

