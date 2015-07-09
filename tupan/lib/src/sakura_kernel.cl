#include "sakura_kernel_common.h"


kernel void
sakura_kernel(
	uint_t const ni,
	global real_t1 const __im[restrict],
	global real_t1 const __irx[restrict],
	global real_t1 const __iry[restrict],
	global real_t1 const __irz[restrict],
	global real_t1 const __ie2[restrict],
	global real_t1 const __ivx[restrict],
	global real_t1 const __ivy[restrict],
	global real_t1 const __ivz[restrict],
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
	global real_t1 __idrx[restrict],
	global real_t1 __idry[restrict],
	global real_t1 __idrz[restrict],
	global real_t1 __idvx[restrict],
	global real_t1 __idvy[restrict],
	global real_t1 __idvz[restrict])
{
	uint_t lid = get_local_id(0);
	uint_t gid = get_global_id(0);
	uint_t i = gid % ni;

	real_t1 im = __im[i];
	real_t1 irx = __irx[i];
	real_t1 iry = __iry[i];
	real_t1 irz = __irz[i];
	real_t1 ie2 = __ie2[i];
	real_t1 ivx = __ivx[i];
	real_t1 ivy = __ivy[i];
	real_t1 ivz = __ivz[i];

	real_t1 idrx = (real_t1)(0);
	real_t1 idry = (real_t1)(0);
	real_t1 idrz = (real_t1)(0);
	real_t1 idvx = (real_t1)(0);
	real_t1 idvy = (real_t1)(0);
	real_t1 idvz = (real_t1)(0);

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
			sakura_kernel_core(
				dt, flag,
				im, irx, iry, irz, ie2, ivx, ivy, ivz,
				jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
				&idrx, &idry, &idrz, &idvx, &idvy, &idvz);
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
		real_t jvx = __jvx[k];
		real_t jvy = __jvy[k];
		real_t jvz = __jvz[k];
		sakura_kernel_core(
			dt, flag,
			im, irx, iry, irz, ie2, ivx, ivy, ivz,
			jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
			&idrx, &idry, &idrz, &idvx, &idvy, &idvz);
	}

	__idrx[i] = idrx;
	__idry[i] = idry;
	__idrz[i] = idrz;
	__idvx[i] = idvx;
	__idvy[i] = idvy;
	__idvz[i] = idvz;
}

