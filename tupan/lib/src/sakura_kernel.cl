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

	Sakura_IData ip = (Sakura_IData){
		.drx = 0,
		.dry = 0,
		.drz = 0,
		.dvx = 0,
		.dvy = 0,
		.dvz = 0,
		.rx = __irx[i],
		.ry = __iry[i],
		.rz = __irz[i],
		.vx = __ivx[i],
		.vy = __ivy[i],
		.vz = __ivz[i],
		.e2 = __ie2[i],
		.m = __im[i],
	};

	uint_t j = 0;

	#ifdef FAST_LOCAL_MEM
	local Sakura_JData _jp[LSIZE];
	#pragma unroll
	for (; (j + LSIZE - 1) < nj; j += LSIZE) {
		barrier(CLK_LOCAL_MEM_FENCE);
		_jp[lid] = (Sakura_JData){
			.rx = __jrx[j + lid],
			.ry = __jry[j + lid],
			.rz = __jrz[j + lid],
			.vx = __jvx[j + lid],
			.vy = __jvy[j + lid],
			.vz = __jvz[j + lid],
			.e2 = __je2[j + lid],
			.m = __jm[j + lid],
		};
		barrier(CLK_LOCAL_MEM_FENCE);
		#pragma unroll
		for (uint_t k = 0; k < LSIZE; ++k) {
			ip = sakura_kernel_core(ip, _jp[k], dt, flag);
		}
	}
	#endif

	#pragma unroll
	for (uint_t k = j; k < nj; ++k) {
		Sakura_JData jp = (Sakura_JData){
			.rx = __jrx[k],
			.ry = __jry[k],
			.rz = __jrz[k],
			.vx = __jvx[k],
			.vy = __jvy[k],
			.vz = __jvz[k],
			.e2 = __je2[k],
			.m = __jm[k],
		};
		ip = sakura_kernel_core(ip, jp, dt, flag);
	}

	__idrx[i] = ip.drx;
	__idry[i] = ip.dry;
	__idrz[i] = ip.drz;
	__idvx[i] = ip.dvx;
	__idvy[i] = ip.dvy;
	__idvz[i] = ip.dvz;
}

