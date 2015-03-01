#include "nreg_kernels_common.h"


__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
kernel void
nreg_Xkernel(
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
	real_t const dt,
	global real_tnxm __idrx[restrict],
	global real_tnxm __idry[restrict],
	global real_tnxm __idrz[restrict],
	global real_tnxm __iax[restrict],
	global real_tnxm __iay[restrict],
	global real_tnxm __iaz[restrict],
	global real_tnxm __iu[restrict])
{
	uint_t lid = get_local_id(0);
	uint_t gid = get_global_id(0);
	gid = (gid < ni) ? (gid):(0);

	real_tnxm im; icopy(im, __im[gid]);
	real_tnxm irx; icopy(irx, __irx[gid]);
	real_tnxm iry; icopy(iry, __iry[gid]);
	real_tnxm irz; icopy(irz, __irz[gid]);
	real_tnxm ie2; icopy(ie2, __ie2[gid]);
	real_tnxm ivx; icopy(ivx, __ivx[gid]);
	real_tnxm ivy; icopy(ivy, __ivy[gid]);
	real_tnxm ivz; icopy(ivz, __ivz[gid]);

	real_tnxm idrx = {0};
	real_tnxm idry = {0};
	real_tnxm idrz = {0};
	real_tnxm iax = {0};
	real_tnxm iay = {0};
	real_tnxm iaz = {0};
	real_tnxm iu = {0};

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
				nreg_Xkernel_core(
					dt,
					im[i], irx[i], iry[i], irz[i],
					ie2[i], ivx[i], ivy[i], ivz[i],
					jm, jrx, jry, jrz,
					je2, jvx, jvy, jvz,
					&idrx[i], &idry[i], &idrz[i],
					&iax[i], &iay[i], &iaz[i], &iu[i]);
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
			nreg_Xkernel_core(
				dt,
				im[i], irx[i], iry[i], irz[i],
				ie2[i], ivx[i], ivy[i], ivz[i],
				jm, jrx, jry, jrz,
				je2, jvx, jvy, jvz,
				&idrx[i], &idry[i], &idrz[i],
				&iax[i], &iay[i], &iaz[i], &iu[i]);
		}
	}

	#pragma unroll
	for (uint_t i = 0; i < IUNROLL; ++i) {
		iu[i] *= im[i];
	}

	icopy(__idrx[gid], idrx);
	icopy(__idry[gid], idry);
	icopy(__idrz[gid], idrz);
	icopy(__iax[gid], iax);
	icopy(__iay[gid], iay);
	icopy(__iaz[gid], iaz);
	icopy(__iu[gid], iu);
}


__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
kernel void
nreg_Vkernel(
	uint_t const ni,
	global real_tnxm const __im[restrict],
	global real_tnxm const __ivx[restrict],
	global real_tnxm const __ivy[restrict],
	global real_tnxm const __ivz[restrict],
	global real_tnxm const __iax[restrict],
	global real_tnxm const __iay[restrict],
	global real_tnxm const __iaz[restrict],
	uint_t const nj,
	global real_t const __jm[restrict],
	global real_t const __jvx[restrict],
	global real_t const __jvy[restrict],
	global real_t const __jvz[restrict],
	global real_t const __jax[restrict],
	global real_t const __jay[restrict],
	global real_t const __jaz[restrict],
	real_t const dt,
	global real_tnxm __idvx[restrict],
	global real_tnxm __idvy[restrict],
	global real_tnxm __idvz[restrict],
	global real_tnxm __ik[restrict])
{
	uint_t lid = get_local_id(0);
	uint_t gid = get_global_id(0);
	gid = (gid < ni) ? (gid):(0);

	real_tnxm im; icopy(im, __im[gid]);
	real_tnxm ivx; icopy(ivx, __ivx[gid]);
	real_tnxm ivy; icopy(ivy, __ivy[gid]);
	real_tnxm ivz; icopy(ivz, __ivz[gid]);
	real_tnxm iax; icopy(iax, __iax[gid]);
	real_tnxm iay; icopy(iay, __iay[gid]);
	real_tnxm iaz; icopy(iaz, __iaz[gid]);

	real_tnxm idvx = {0};
	real_tnxm idvy = {0};
	real_tnxm idvz = {0};
	real_tnxm ik = {0};

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
			#pragma unroll
			for (uint_t i = 0; i < IUNROLL; ++i) {
				nreg_Vkernel_core(
					dt,
					im[i], ivx[i], ivy[i], ivz[i],
					iax[i], iay[i], iaz[i],
					jm, jvx, jvy, jvz,
					jax, jay, jaz,
					&idvx[i], &idvy[i], &idvz[i], &ik[i]);
			}
		}
	}
	#endif

	#pragma unroll
	for (; j < nj; ++j) {
		real_t jm = __jm[j];
		real_t jvx = __jvx[j];
		real_t jvy = __jvy[j];
		real_t jvz = __jvz[j];
		real_t jax = __jax[j];
		real_t jay = __jay[j];
		real_t jaz = __jaz[j];
		#pragma unroll
		for (uint_t i = 0; i < IUNROLL; ++i) {
			nreg_Vkernel_core(
				dt,
				im[i], ivx[i], ivy[i], ivz[i],
				iax[i], iay[i], iaz[i],
				jm, jvx, jvy, jvz,
				jax, jay, jaz,
				&idvx[i], &idvy[i], &idvz[i], &ik[i]);
		}
	}

	#pragma unroll
	for (uint_t i = 0; i < IUNROLL; ++i) {
		ik[i] *= im[i];
	}

	icopy(__idvx[gid], idvx);
	icopy(__idvy[gid], idvy);
	icopy(__idvz[gid], idvz);
	icopy(__ik[gid], ik);
}

