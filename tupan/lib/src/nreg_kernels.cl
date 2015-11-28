#include "nreg_kernels_common.h"


kernel void
nreg_Xkernel(
	const uint_t ni,
	global const real_tn __im[],
	global const real_tn __irx[],
	global const real_tn __iry[],
	global const real_tn __irz[],
	global const real_tn __ie2[],
	global const real_tn __ivx[],
	global const real_tn __ivy[],
	global const real_tn __ivz[],
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
	global real_tn __idrx[],
	global real_tn __idry[],
	global real_tn __idrz[],
	global real_tn __iax[],
	global real_tn __iay[],
	global real_tn __iaz[],
	global real_tn __iu[])
{
	uint_t lid = get_local_id(0);
	uint_t gid = get_global_id(0);
	uint_t i = gid % ni;

	Nreg_X_IData ip = (Nreg_X_IData){
		.drx = 0,
		.dry = 0,
		.drz = 0,
		.ax = 0,
		.ay = 0,
		.az = 0,
		.u = 0,
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
	local Nreg_X_JData _jp[LSIZE];
	#pragma unroll
	for (; (j + LSIZE - 1) < nj; j += LSIZE) {
		barrier(CLK_LOCAL_MEM_FENCE);
		_jp[lid] = (Nreg_X_JData){
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
			ip = nreg_Xkernel_core(ip, _jp[k], dt);
		}
	}
	#endif

	#pragma unroll
	for (uint_t k = j; k < nj; ++k) {
		Nreg_X_JData jp = (Nreg_X_JData){
			.rx = __jrx[k],
			.ry = __jry[k],
			.rz = __jrz[k],
			.vx = __jvx[k],
			.vy = __jvy[k],
			.vz = __jvz[k],
			.e2 = __je2[k],
			.m = __jm[k],
		};
		ip = nreg_Xkernel_core(ip, jp, dt);
	}

	__idrx[i] = ip.drx;
	__idry[i] = ip.dry;
	__idrz[i] = ip.drz;
	__iax[i] = ip.ax;
	__iay[i] = ip.ay;
	__iaz[i] = ip.az;
	__iu[i] = ip.m * ip.u;
}


kernel void
nreg_Vkernel(
	const uint_t ni,
	global const real_tn __im[],
	global const real_tn __ivx[],
	global const real_tn __ivy[],
	global const real_tn __ivz[],
	global const real_tn __iax[],
	global const real_tn __iay[],
	global const real_tn __iaz[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __jvx[],
	global const real_t __jvy[],
	global const real_t __jvz[],
	global const real_t __jax[],
	global const real_t __jay[],
	global const real_t __jaz[],
	const real_t dt,
	global real_tn __idvx[],
	global real_tn __idvy[],
	global real_tn __idvz[],
	global real_tn __ik[])
{
	uint_t lid = get_local_id(0);
	uint_t gid = get_global_id(0);
	uint_t i = gid % ni;

	Nreg_V_IData ip = (Nreg_V_IData){
		.dvx = 0,
		.dvy = 0,
		.dvz = 0,
		.k = 0,
		.vx = __ivx[i],
		.vy = __ivy[i],
		.vz = __ivz[i],
		.ax = __iax[i],
		.ay = __iay[i],
		.az = __iaz[i],
		.m = __im[i],
	};

	uint_t j = 0;

	#ifdef FAST_LOCAL_MEM
	local Nreg_V_JData _jp[LSIZE];
	#pragma unroll
	for (; (j + LSIZE - 1) < nj; j += LSIZE) {
		barrier(CLK_LOCAL_MEM_FENCE);
		_jp[lid] = (Nreg_V_JData){
			.vx = __jvx[j + lid],
			.vy = __jvy[j + lid],
			.vz = __jvz[j + lid],
			.ax = __jax[j + lid],
			.ay = __jay[j + lid],
			.az = __jaz[j + lid],
			.m = __jm[j + lid],
		};
		barrier(CLK_LOCAL_MEM_FENCE);
		#pragma unroll
		for (uint_t k = 0; k < LSIZE; ++k) {
			ip = nreg_Vkernel_core(ip, _jp[k], dt);
		}
	}
	#endif

	#pragma unroll
	for (uint_t k = j; k < nj; ++k) {
		Nreg_V_JData jp = (Nreg_V_JData){
			.vx = __jvx[k],
			.vy = __jvy[k],
			.vz = __jvz[k],
			.ax = __jax[k],
			.ay = __jay[k],
			.az = __jaz[k],
			.m = __jm[k],
		};
		ip = nreg_Vkernel_core(ip, jp, dt);
	}

	__idvx[i] = ip.dvx;
	__idvy[i] = ip.dvy;
	__idvz[i] = ip.dvz;
	__ik[i] = ip.m * ip.k;
}

