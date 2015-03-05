#include "sakura_kernel_common.h"


__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
kernel void
sakura_kernel(
	uint_t const ni,
	global real_t const __im[restrict],
	global real_t const __irx[restrict],
	global real_t const __iry[restrict],
	global real_t const __irz[restrict],
	global real_t const __ie2[restrict],
	global real_t const __ivx[restrict],
	global real_t const __ivy[restrict],
	global real_t const __ivz[restrict],
	uint_t const nj,
	global real_t const __jm[restrict],
	global real_t const __jrx[restrict],
	global real_t const __jry[restrict],
	global real_t const __jrz[restrict],
	global real_t const __je2[restrict],
	global real_t const __jvx[restrict],
	global real_t const __jvy[restrict],
	global real_t const __jvz[restrict],
	real_t const dt,
	int_t const flag,
	global real_t __idrx[restrict],
	global real_t __idry[restrict],
	global real_t __idrz[restrict],
	global real_t __idvx[restrict],
	global real_t __idvy[restrict],
	global real_t __idvz[restrict])
{
	uint_t lid = get_local_id(0);
	uint_t gid = get_global_id(0);
	gid = (gid < ni) ? (gid):(0);

	real_t im[] = aloadn(gid, __im);
	real_t irx[] = aloadn(gid, __irx);
	real_t iry[] = aloadn(gid, __iry);
	real_t irz[] = aloadn(gid, __irz);
	real_t ie2[] = aloadn(gid, __ie2);
	real_t ivx[] = aloadn(gid, __ivx);
	real_t ivy[] = aloadn(gid, __ivy);
	real_t ivz[] = aloadn(gid, __ivz);

	real_t idrx[IUNROLL] = {0};
	real_t idry[IUNROLL] = {0};
	real_t idrz[IUNROLL] = {0};
	real_t idvx[IUNROLL] = {0};
	real_t idvy[IUNROLL] = {0};
	real_t idvz[IUNROLL] = {0};

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
				sakura_kernel_core(
					dt, flag,
					im[i], irx[i], iry[i], irz[i],
					ie2[i], ivx[i], ivy[i], ivz[i],
					jm, jrx, jry, jrz,
					je2, jvx, jvy, jvz,
					&idrx[i], &idry[i], &idrz[i],
					&idvx[i], &idvy[i], &idvz[i]);
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
			sakura_kernel_core(
				dt, flag,
				im[i], irx[i], iry[i], irz[i],
				ie2[i], ivx[i], ivy[i], ivz[i],
				jm, jrx, jry, jrz,
				je2, jvx, jvy, jvz,
				&idrx[i], &idry[i], &idrz[i],
				&idvx[i], &idvy[i], &idvz[i]);
		}
	}

	astoren(idrx, gid, __idrx);
	astoren(idry, gid, __idry);
	astoren(idrz, gid, __idrz);
	astoren(idvx, gid, __idvx);
	astoren(idvy, gid, __idvy);
	astoren(idvz, gid, __idvz);
}

