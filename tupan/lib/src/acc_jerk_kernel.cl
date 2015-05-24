#include "acc_jerk_kernel_common.h"


kernel void
acc_jerk_kernel(
	uint_t const ni,
	global real_tnxm const __im[restrict],
	global real_tnxm const __irx[restrict],
	global real_tnxm const __iry[restrict],
	global real_tnxm const __irz[restrict],
	global real_tnxm const __ie2[restrict],
	global real_tnxm const __ivx[restrict],
	global real_tnxm const __ivy[restrict],
	global real_tnxm const __ivz[restrict],
	uint_t const nj,
	global real_t const __jm[restrict],
	global real_t const __jrx[restrict],
	global real_t const __jry[restrict],
	global real_t const __jrz[restrict],
	global real_t const __je2[restrict],
	global real_t const __jvx[restrict],
	global real_t const __jvy[restrict],
	global real_t const __jvz[restrict],
	global real_tnxm __iax[restrict],
	global real_tnxm __iay[restrict],
	global real_tnxm __iaz[restrict],
	global real_tnxm __ijx[restrict],
	global real_tnxm __ijy[restrict],
	global real_tnxm __ijz[restrict])
{
	uint_t lid = get_local_id(0);
	uint_t gid = get_global_id(0);
	gid %= ni;

	real_tnxm im = aloadn(0, __im[gid]);
	real_tnxm irx = aloadn(0, __irx[gid]);
	real_tnxm iry = aloadn(0, __iry[gid]);
	real_tnxm irz = aloadn(0, __irz[gid]);
	real_tnxm ie2 = aloadn(0, __ie2[gid]);
	real_tnxm ivx = aloadn(0, __ivx[gid]);
	real_tnxm ivy = aloadn(0, __ivy[gid]);
	real_tnxm ivz = aloadn(0, __ivz[gid]);

	real_tnxm iax = {(real_tn)(0)};
	real_tnxm iay = {(real_tn)(0)};
	real_tnxm iaz = {(real_tn)(0)};
	real_tnxm ijx = {(real_tn)(0)};
	real_tnxm ijy = {(real_tn)(0)};
	real_tnxm ijz = {(real_tn)(0)};

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

	astoren(iax, 0, __iax[gid]);
	astoren(iay, 0, __iay[gid]);
	astoren(iaz, 0, __iaz[gid]);
	astoren(ijx, 0, __ijx[gid]);
	astoren(ijy, 0, __ijy[gid]);
	astoren(ijz, 0, __ijz[gid]);
}

