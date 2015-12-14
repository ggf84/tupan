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
	uint_t start = get_group_id(0) * get_local_size(0);
	uint_t stride = get_num_groups(0) * get_local_size(0);
	for (uint_t ii = start; ii * SIMD < ni; ii += stride) {
		uint_t i = ii + lid;
		i *= (i * SIMD < ni);

		vec(Nreg_X_Data) ip = (vec(Nreg_X_Data)){
			.drx = (real_tn)(0),
			.dry = (real_tn)(0),
			.drz = (real_tn)(0),
			.ax = (real_tn)(0),
			.ay = (real_tn)(0),
			.az = (real_tn)(0),
			.u = (real_tn)(0),
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
			Nreg_X_Data jp = (Nreg_X_Data){
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
			local Nreg_X_Data _jp[LSIZE];
			_jp[lid] = jp;
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll
			for (uint_t k = 0; k < LSIZE; ++k) {
				ip = nreg_Xkernel_core(ip, _jp[k], dt);
			}
		}
		#endif

		for (; ((j + 1) - 1) < nj; j += 1) {
			Nreg_X_Data jp = (Nreg_X_Data){
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.e2 = __je2[j],
				.m = __jm[j],
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
	uint_t start = get_group_id(0) * get_local_size(0);
	uint_t stride = get_num_groups(0) * get_local_size(0);
	for (uint_t ii = start; ii * SIMD < ni; ii += stride) {
		uint_t i = ii + lid;
		i *= (i * SIMD < ni);

		vec(Nreg_V_Data) ip = (vec(Nreg_V_Data)){
			.dvx = (real_tn)(0),
			.dvy = (real_tn)(0),
			.dvz = (real_tn)(0),
			.k = (real_tn)(0),
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
		for (; ((j + LSIZE) - 1) < nj; j += LSIZE) {
			Nreg_V_Data jp = (Nreg_V_Data){
				.vx = __jvx[j + lid],
				.vy = __jvy[j + lid],
				.vz = __jvz[j + lid],
				.ax = __jax[j + lid],
				.ay = __jay[j + lid],
				.az = __jaz[j + lid],
				.m = __jm[j + lid],
			};
			barrier(CLK_LOCAL_MEM_FENCE);
			local Nreg_V_Data _jp[LSIZE];
			_jp[lid] = jp;
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll
			for (uint_t k = 0; k < LSIZE; ++k) {
				ip = nreg_Vkernel_core(ip, _jp[k], dt);
			}
		}
		#endif

		for (; ((j + 1) - 1) < nj; j += 1) {
			Nreg_V_Data jp = (Nreg_V_Data){
				.vx = __jvx[j],
				.vy = __jvy[j],
				.vz = __jvz[j],
				.ax = __jax[j],
				.ay = __jay[j],
				.az = __jaz[j],
				.m = __jm[j],
			};
			ip = nreg_Vkernel_core(ip, jp, dt);
		}

		__idvx[i] = ip.dvx;
		__idvy[i] = ip.dvy;
		__idvz[i] = ip.dvz;
		__ik[i] = ip.m * ip.k;
	}
}

