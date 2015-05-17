#include "acc_jerk_kernel_common.h"


kernel void
acc_jerk_kernel(
	uint_t const ni,
	global real_tn const __im[restrict],
	global real_tn const __irx[restrict],
	global real_tn const __iry[restrict],
	global real_tn const __irz[restrict],
	global real_tn const __ie2[restrict],
	global real_tn const __ivx[restrict],
	global real_tn const __ivy[restrict],
	global real_tn const __ivz[restrict],
	uint_t const nj,
	global real_t const __jm[restrict],
	global real_t const __jrx[restrict],
	global real_t const __jry[restrict],
	global real_t const __jrz[restrict],
	global real_t const __je2[restrict],
	global real_t const __jvx[restrict],
	global real_t const __jvy[restrict],
	global real_t const __jvz[restrict],
	global real_tn __iax[restrict],
	global real_tn __iay[restrict],
	global real_tn __iaz[restrict],
	global real_tn __ijx[restrict],
	global real_tn __ijy[restrict],
	global real_tn __ijz[restrict])
{
	uint_t lid = get_local_id(0);
	uint_t gid = get_global_id(0);
	gid %= ni;

	real_tn im[] = aloadn(gid, __im);
	real_tn irx[] = aloadn(gid, __irx);
	real_tn iry[] = aloadn(gid, __iry);
	real_tn irz[] = aloadn(gid, __irz);
	real_tn ie2[] = aloadn(gid, __ie2);
	real_tn ivx[] = aloadn(gid, __ivx);
	real_tn ivy[] = aloadn(gid, __ivy);
	real_tn ivz[] = aloadn(gid, __ivz);

	real_tn iax[IUNROLL] = {(real_tn)(0)};
	real_tn iay[IUNROLL] = {(real_tn)(0)};
	real_tn iaz[IUNROLL] = {(real_tn)(0)};
	real_tn ijx[IUNROLL] = {(real_tn)(0)};
	real_tn ijy[IUNROLL] = {(real_tn)(0)};
	real_tn ijz[IUNROLL] = {(real_tn)(0)};

	uint_t j = 0;

	#ifdef FAST_LOCAL_MEM
	local real_t _jm[LSIZE];
	local real_t _jrx[LSIZE];
	local real_t _jry[LSIZE];
	local real_t _jrz[LSIZE];
	local real_t _je2[LSIZE];
	local real_t _jvx[LSIZE];
	local real_t _jvy[LSIZE];
	local real_t _jvz[LSIZE];
	#pragma unroll
	for (; (j + LSIZE - 1) < nj; j += LSIZE) {
		barrier(CLK_LOCAL_MEM_FENCE);
		_jm[lid] = __jm[j + lid];
		_jrx[lid] = __jrx[j + lid];
		_jry[lid] = __jry[j + lid];
		_jrz[lid] = __jrz[j + lid];
		_je2[lid] = __je2[j + lid];
		_jvx[lid] = __jvx[j + lid];
		_jvy[lid] = __jvy[j + lid];
		_jvz[lid] = __jvz[j + lid];
		barrier(CLK_LOCAL_MEM_FENCE);
		#pragma unroll
		for (uint_t k = 0; k < LSIZE; ++k) {
			real_t jm = _jm[k];
			real_t jrx = _jrx[k];
			real_t jry = _jry[k];
			real_t jrz = _jrz[k];
			real_t je2 = _je2[k];
			real_t jvx = _jvx[k];
			real_t jvy = _jvy[k];
			real_t jvz = _jvz[k];
			#pragma unroll
			for (uint_t i = 0; i < IUNROLL; ++i) {
				acc_jerk_kernel_core(
					im[i], irx[i], iry[i], irz[i],
					ie2[i], ivx[i], ivy[i], ivz[i],
					jm, jrx, jry, jrz,
					je2, jvx, jvy, jvz,
					&iax[i], &iay[i], &iaz[i],
					&ijx[i], &ijy[i], &ijz[i]);
			}
		}
	}
	#endif

	#pragma unroll
	for (; j < nj; ++j) {
		real_t jm = __jm[j];
		real_t jrx = __jrx[j];
		real_t jry = __jry[j];
		real_t jrz = __jrz[j];
		real_t je2 = __je2[j];
		real_t jvx = __jvx[j];
		real_t jvy = __jvy[j];
		real_t jvz = __jvz[j];
		#pragma unroll
		for (uint_t i = 0; i < IUNROLL; ++i) {
			acc_jerk_kernel_core(
				im[i], irx[i], iry[i], irz[i],
				ie2[i], ivx[i], ivy[i], ivz[i],
				jm, jrx, jry, jrz,
				je2, jvx, jvy, jvz,
				&iax[i], &iay[i], &iaz[i],
				&ijx[i], &ijy[i], &ijz[i]);
		}
	}

	astoren(iax, gid, __iax);
	astoren(iay, gid, __iay);
	astoren(iaz, gid, __iaz);
	astoren(ijx, gid, __ijx);
	astoren(ijy, gid, __ijy);
	astoren(ijz, gid, __ijz);
}

