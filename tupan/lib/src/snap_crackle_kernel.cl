#include "snap_crackle_kernel_common.h"


__attribute__((vec_type_hint(real_tn)))
kernel void
snap_crackle_kernel(
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
	gid %= ni;

	real_tn im[] = aloadn(gid, __im);
	real_tn irx[] = aloadn(gid, __irx);
	real_tn iry[] = aloadn(gid, __iry);
	real_tn irz[] = aloadn(gid, __irz);
	real_tn ie2[] = aloadn(gid, __ie2);
	real_tn ivx[] = aloadn(gid, __ivx);
	real_tn ivy[] = aloadn(gid, __ivy);
	real_tn ivz[] = aloadn(gid, __ivz);
	real_tn iax[] = aloadn(gid, __iax);
	real_tn iay[] = aloadn(gid, __iay);
	real_tn iaz[] = aloadn(gid, __iaz);
	real_tn ijx[] = aloadn(gid, __ijx);
	real_tn ijy[] = aloadn(gid, __ijy);
	real_tn ijz[] = aloadn(gid, __ijz);

	real_tn isx[IUNROLL] = {(real_tn)(0)};
	real_tn isy[IUNROLL] = {(real_tn)(0)};
	real_tn isz[IUNROLL] = {(real_tn)(0)};
	real_tn icx[IUNROLL] = {(real_tn)(0)};
	real_tn icy[IUNROLL] = {(real_tn)(0)};
	real_tn icz[IUNROLL] = {(real_tn)(0)};

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
			#pragma unroll
			for (uint_t i = 0; i < IUNROLL; ++i) {
				snap_crackle_kernel_core(
					im[i], irx[i], iry[i], irz[i],
					ie2[i], ivx[i], ivy[i], ivz[i],
					iax[i], iay[i], iaz[i],
					ijx[i], ijy[i], ijz[i],
					jm, jrx, jry, jrz,
					je2, jvx, jvy, jvz,
					jax, jay, jaz,
					jjx, jjy, jjz,
					&isx[i], &isy[i], &isz[i],
					&icx[i], &icy[i], &icz[i]);
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
		real_t jax = __jax[j];
		real_t jay = __jay[j];
		real_t jaz = __jaz[j];
		real_t jjx = __jjx[j];
		real_t jjy = __jjy[j];
		real_t jjz = __jjz[j];
		#pragma unroll
		for (uint_t i = 0; i < IUNROLL; ++i) {
			snap_crackle_kernel_core(
				im[i], irx[i], iry[i], irz[i],
				ie2[i], ivx[i], ivy[i], ivz[i],
				iax[i], iay[i], iaz[i],
				ijx[i], ijy[i], ijz[i],
				jm, jrx, jry, jrz,
				je2, jvx, jvy, jvz,
				jax, jay, jaz,
				jjx, jjy, jjz,
				&isx[i], &isy[i], &isz[i],
				&icx[i], &icy[i], &icz[i]);
		}
	}

	astoren(isx, gid, __isx);
	astoren(isy, gid, __isy);
	astoren(isz, gid, __isz);
	astoren(icx, gid, __icx);
	astoren(icy, gid, __icy);
	astoren(icz, gid, __icz);
}

