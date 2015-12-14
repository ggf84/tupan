#include "sakura_kernel_common.h"


kernel void
sakura_kernel(
	const uint_t ni,
	global real_t1 const __im[],
	global real_t1 const __irx[],
	global real_t1 const __iry[],
	global real_t1 const __irz[],
	global real_t1 const __ie2[],
	global real_t1 const __ivx[],
	global real_t1 const __ivy[],
	global real_t1 const __ivz[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __jrx[],
	global const real_t __jry[],
	global const real_t __jrz[],
	global const real_t __je2[],
	global const real_t __jvx[],
	global const real_t __jvy[],
	global const real_t __jvz[],
	const real_t dt,
	const int_t flag,
	global real_t1 __idrx[],
	global real_t1 __idry[],
	global real_t1 __idrz[],
	global real_t1 __idvx[],
	global real_t1 __idvy[],
	global real_t1 __idvz[])
{
	uint_t lid = get_local_id(0);
	uint_t start = get_group_id(0) * get_local_size(0);
	uint_t stride = get_num_groups(0) * get_local_size(0);
	for (uint_t ii = start; ii * 1 < ni; ii += stride) {
		uint_t i = ii + lid;
		i *= (i * 1 < ni);

		Sakura_Data1 ip = (Sakura_Data1){
			.drx = (real_t1)(0),
			.dry = (real_t1)(0),
			.drz = (real_t1)(0),
			.dvx = (real_t1)(0),
			.dvy = (real_t1)(0),
			.dvz = (real_t1)(0),
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
		for (; ((j + LSIZE) - 1) < nj; j += LSIZE) {
			Sakura_Data jp = (Sakura_Data){
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
			local Sakura_Data _jp[LSIZE];
			_jp[lid] = jp;
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll
			for (uint_t k = 0; k < LSIZE; ++k) {
				ip = sakura_kernel_core(ip, _jp[k], dt, flag);
			}
		}
		#endif

		for (; ((j + 1) - 1) < nj; j += 1) {
			Sakura_Data jp = (Sakura_Data){
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.e2 = __je2[j],
				.m = __jm[j],
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
}

