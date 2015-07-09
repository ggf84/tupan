#include "snp_crk_kernel_common.h"


kernel void
snp_crk_kernel(
	uint_t const ni,
	global real_tn const __im[restrict],
	global real_tn const __irx[restrict],
	global real_tn const __iry[restrict],
	global real_tn const __irz[restrict],
	global real_tn const __ie2[restrict],
	global real_tn const __ivx[restrict],
	global real_tn const __ivy[restrict],
	global real_tn const __ivz[restrict],
	global real_tn const __iax[restrict],
	global real_tn const __iay[restrict],
	global real_tn const __iaz[restrict],
	global real_tn const __ijx[restrict],
	global real_tn const __ijy[restrict],
	global real_tn const __ijz[restrict],
	uint_t const nj,
	global real_t const __jm[restrict],
	global real_t const __jrx[restrict],
	global real_t const __jry[restrict],
	global real_t const __jrz[restrict],
	global real_t const __je2[restrict],
	global real_t const __jvx[restrict],
	global real_t const __jvy[restrict],
	global real_t const __jvz[restrict],
	global real_t const __jax[restrict],
	global real_t const __jay[restrict],
	global real_t const __jaz[restrict],
	global real_t const __jjx[restrict],
	global real_t const __jjy[restrict],
	global real_t const __jjz[restrict],
	global real_tn __isx[restrict],
	global real_tn __isy[restrict],
	global real_tn __isz[restrict],
	global real_tn __icx[restrict],
	global real_tn __icy[restrict],
	global real_tn __icz[restrict])
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
	real_tn iax = __iax[i];
	real_tn iay = __iay[i];
	real_tn iaz = __iaz[i];
	real_tn ijx = __ijx[i];
	real_tn ijy = __ijy[i];
	real_tn ijz = __ijz[i];

	real_tn isx = (real_tn)(0);
	real_tn isy = (real_tn)(0);
	real_tn isz = (real_tn)(0);
	real_tn icx = (real_tn)(0);
	real_tn icy = (real_tn)(0);
	real_tn icz = (real_tn)(0);

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
	local real_t _jax[LSIZE];
	local real_t _jay[LSIZE];
	local real_t _jaz[LSIZE];
	local real_t _jjx[LSIZE];
	local real_t _jjy[LSIZE];
	local real_t _jjz[LSIZE];
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
		_jax[lid] = __jax[j + lid];
		_jay[lid] = __jay[j + lid];
		_jaz[lid] = __jaz[j + lid];
		_jjx[lid] = __jjx[j + lid];
		_jjy[lid] = __jjy[j + lid];
		_jjz[lid] = __jjz[j + lid];
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
			real_t jax = _jax[k];
			real_t jay = _jay[k];
			real_t jaz = _jaz[k];
			real_t jjx = _jjx[k];
			real_t jjy = _jjy[k];
			real_t jjz = _jjz[k];
			snp_crk_kernel_core(
				im, irx, iry, irz, ie2, ivx, ivy, ivz,
				iax, iay, iaz, ijx, ijy, ijz,
				jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
				jax, jay, jaz, jjx, jjy, jjz,
				&isx, &isy, &isz, &icx, &icy, &icz);
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
		real_t jax = __jax[k];
		real_t jay = __jay[k];
		real_t jaz = __jaz[k];
		real_t jjx = __jjx[k];
		real_t jjy = __jjy[k];
		real_t jjz = __jjz[k];
		snp_crk_kernel_core(
			im, irx, iry, irz, ie2, ivx, ivy, ivz,
			iax, iay, iaz, ijx, ijy, ijz,
			jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
			jax, jay, jaz, jjx, jjy, jjz,
			&isx, &isy, &isz, &icx, &icy, &icz);
	}

	__isx[i] = isx;
	__isy[i] = isy;
	__isz[i] = isz;
	__icx[i] = icx;
	__icy[i] = icy;
	__icz[i] = icz;
}

