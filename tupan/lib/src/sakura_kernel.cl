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

	real_t im = __im[gid];
	real_t irx = __irx[gid];
	real_t iry = __iry[gid];
	real_t irz = __irz[gid];
	real_t ie2 = __ie2[gid];
	real_t ivx = __ivx[gid];
	real_t ivy = __ivy[gid];
	real_t ivz = __ivz[gid];

	real_t idrx = (real_t)(0);
	real_t idry = (real_t)(0);
	real_t idrz = (real_t)(0);
	real_t idvx = (real_t)(0);
	real_t idvy = (real_t)(0);
	real_t idvz = (real_t)(0);

	uint_t j = 0;

	#ifdef FAST_LOCAL_MEM
	local real_t _jm[2][LSIZE];
	local real_t _jrx[2][LSIZE];
	local real_t _jry[2][LSIZE];
	local real_t _jrz[2][LSIZE];
	local real_t _je2[2][LSIZE];
	local real_t _jvx[2][LSIZE];
	local real_t _jvy[2][LSIZE];
	local real_t _jvz[2][LSIZE];
	#pragma unroll
	for (uint_t g = GROUPS; g > 0; g >>= 1) {
		#pragma unroll
		for (; (j + g * LSIZE - 1) < nj; j += g * LSIZE) {
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll
			for (uint_t k = 0; k < g; ++k) {
				_jm[k & 1][lid] = __jm[j + k * LSIZE + lid];
				_jrx[k & 1][lid] = __jrx[j + k * LSIZE + lid];
				_jry[k & 1][lid] = __jry[j + k * LSIZE + lid];
				_jrz[k & 1][lid] = __jrz[j + k * LSIZE + lid];
				_je2[k & 1][lid] = __je2[j + k * LSIZE + lid];
				_jvx[k & 1][lid] = __jvx[j + k * LSIZE + lid];
				_jvy[k & 1][lid] = __jvy[j + k * LSIZE + lid];
				_jvz[k & 1][lid] = __jvz[j + k * LSIZE + lid];
				barrier(CLK_LOCAL_MEM_FENCE);
				#pragma unroll
				for (uint_t l = 0; l < LSIZE; ++l) {
					real_t jm = _jm[k & 1][l];
					real_t jrx = _jrx[k & 1][l];
					real_t jry = _jry[k & 1][l];
					real_t jrz = _jrz[k & 1][l];
					real_t je2 = _je2[k & 1][l];
					real_t jvx = _jvx[k & 1][l];
					real_t jvy = _jvy[k & 1][l];
					real_t jvz = _jvz[k & 1][l];
					sakura_kernel_core(
						dt, flag,
						im, irx, iry, irz, ie2, ivx, ivy, ivz,
						jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
						&idrx, &idry, &idrz, &idvx, &idvy, &idvz);
				}
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
		sakura_kernel_core(
			dt, flag,
			im, irx, iry, irz, ie2, ivx, ivy, ivz,
			jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
			&idrx, &idry, &idrz, &idvx, &idvy, &idvz);
	}

	__idrx[gid] = idrx;
	__idry[gid] = idry;
	__idrz[gid] = idrz;
	__idvx[gid] = idvx;
	__idvy[gid] = idvy;
	__idvz[gid] = idvz;
}

