#include "nreg_kernels_common.h"


kernel void
nreg_Xkernel(
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
	real_t const dt,
	global real_tn __idrx[restrict],
	global real_tn __idry[restrict],
	global real_tn __idrz[restrict],
	global real_tn __iax[restrict],
	global real_tn __iay[restrict],
	global real_tn __iaz[restrict],
	global real_tn __iu[restrict])
{
	uint_t lid = get_local_id(0);
	uint_t gid = get_global_id(0);
	uint_t i = gid % ni;

	real_tn im = __im[i];
	real_tn irx = __irx[i];
	real_tn iry = __iry[i];
	real_tn irz = __irz[i];
	real_tn ie2 = __ie2[i];
	real_tn ivx = __ivx[i];
	real_tn ivy = __ivy[i];
	real_tn ivz = __ivz[i];

	real_tn idrx = (real_tn)(0);
	real_tn idry = (real_tn)(0);
	real_tn idrz = (real_tn)(0);
	real_tn iax = (real_tn)(0);
	real_tn iay = (real_tn)(0);
	real_tn iaz = (real_tn)(0);
	real_tn iu = (real_tn)(0);

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
			nreg_Xkernel_core(
				dt,
				im, irx, iry, irz, ie2, ivx, ivy, ivz,
				jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
				&idrx, &idry, &idrz, &iax, &iay, &iaz, &iu);
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
		nreg_Xkernel_core(
			dt,
			im, irx, iry, irz, ie2, ivx, ivy, ivz,
			jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
			&idrx, &idry, &idrz, &iax, &iay, &iaz, &iu);
	}

	__idrx[i] = idrx;
	__idry[i] = idry;
	__idrz[i] = idrz;
	__iax[i] = iax;
	__iay[i] = iay;
	__iaz[i] = iaz;
	__iu[i] = im * iu;
}


kernel void
nreg_Vkernel(
	uint_t const ni,
	global real_tn const __im[restrict],
	global real_tn const __ivx[restrict],
	global real_tn const __ivy[restrict],
	global real_tn const __ivz[restrict],
	global real_tn const __iax[restrict],
	global real_tn const __iay[restrict],
	global real_tn const __iaz[restrict],
	uint_t const nj,
	global real_t const __jm[restrict],
	global real_t const __jvx[restrict],
	global real_t const __jvy[restrict],
	global real_t const __jvz[restrict],
	global real_t const __jax[restrict],
	global real_t const __jay[restrict],
	global real_t const __jaz[restrict],
	real_t const dt,
	global real_tn __idvx[restrict],
	global real_tn __idvy[restrict],
	global real_tn __idvz[restrict],
	global real_tn __ik[restrict])
{
	uint_t lid = get_local_id(0);
	uint_t gid = get_global_id(0);
	uint_t i = gid % ni;

	real_tn im = __im[i];
	real_tn ivx = __ivx[i];
	real_tn ivy = __ivy[i];
	real_tn ivz = __ivz[i];
	real_tn iax = __iax[i];
	real_tn iay = __iay[i];
	real_tn iaz = __iaz[i];

	real_tn idvx = (real_tn)(0);
	real_tn idvy = (real_tn)(0);
	real_tn idvz = (real_tn)(0);
	real_tn ik = (real_tn)(0);

	uint_t j = 0;

	#ifdef FAST_LOCAL_MEM
	local real_t _jm[LSIZE];
	local real_t _jvx[LSIZE];
	local real_t _jvy[LSIZE];
	local real_t _jvz[LSIZE];
	local real_t _jax[LSIZE];
	local real_t _jay[LSIZE];
	local real_t _jaz[LSIZE];
	#pragma unroll
	for (; (j + LSIZE - 1) < nj; j += LSIZE) {
		barrier(CLK_LOCAL_MEM_FENCE);
		_jm[lid] = __jm[j + lid];
		_jvx[lid] = __jvx[j + lid];
		_jvy[lid] = __jvy[j + lid];
		_jvz[lid] = __jvz[j + lid];
		_jax[lid] = __jax[j + lid];
		_jay[lid] = __jay[j + lid];
		_jaz[lid] = __jaz[j + lid];
		barrier(CLK_LOCAL_MEM_FENCE);
		#pragma unroll
		for (uint_t k = 0; k < LSIZE; ++k) {
			real_t jm = _jm[k];
			real_t jvx = _jvx[k];
			real_t jvy = _jvy[k];
			real_t jvz = _jvz[k];
			real_t jax = _jax[k];
			real_t jay = _jay[k];
			real_t jaz = _jaz[k];
			nreg_Vkernel_core(
				dt,
				im, ivx, ivy, ivz, iax, iay, iaz,
				jm, jvx, jvy, jvz, jax, jay, jaz,
				&idvx, &idvy, &idvz, &ik);
		}
	}
	#endif

	#pragma unroll
	for (uint_t k = j; k < nj; ++k) {
		real_t jm = __jm[k];
		real_t jvx = __jvx[k];
		real_t jvy = __jvy[k];
		real_t jvz = __jvz[k];
		real_t jax = __jax[k];
		real_t jay = __jay[k];
		real_t jaz = __jaz[k];
		nreg_Vkernel_core(
			dt,
			im, ivx, ivy, ivz, iax, iay, iaz,
			jm, jvx, jvy, jvz, jax, jay, jaz,
			&idvx, &idvy, &idvz, &ik);
	}

	__idvx[i] = idvx;
	__idvy[i] = idvy;
	__idvz[i] = idvz;
	__ik[i] = im * ik;
}

