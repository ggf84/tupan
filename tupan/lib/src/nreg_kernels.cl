#include "nreg_kernels_common.h"


kernel void
nreg_Xkernel(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __irx[],
	global const real_t __iry[],
	global const real_t __irz[],
	global const real_t __ie2[],
	global const real_t __ivx[],
	global const real_t __ivy[],
	global const real_t __ivz[],
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
	global real_t __idrx[],
	global real_t __idry[],
	global real_t __idrz[],
	global real_t __iax[],
	global real_t __iay[],
	global real_t __iaz[],
	global real_t __iu[])
{
	uint_t lid = get_local_id(0);
	uint_t start = get_group_id(0) * get_local_size(0);
	uint_t stride = get_num_groups(0) * get_local_size(0);
	for (uint_t ii = start; ii * SIMD < ni; ii += stride) {
		uint_t i = ii + lid;
		i *= SIMD;
		i = (i+SIMD < ni) ? (i):(ni-SIMD);
		i *= (SIMD < ni);

		vec(Nreg_X_Data) ip = (vec(Nreg_X_Data)){
			.drx = (real_tn)(0),
			.dry = (real_tn)(0),
			.drz = (real_tn)(0),
			.ax = (real_tn)(0),
			.ay = (real_tn)(0),
			.az = (real_tn)(0),
			.u = (real_tn)(0),
			.rx = vec(vload)(0, __irx + i),
			.ry = vec(vload)(0, __iry + i),
			.rz = vec(vload)(0, __irz + i),
			.vx = vec(vload)(0, __ivx + i),
			.vy = vec(vload)(0, __ivy + i),
			.vz = vec(vload)(0, __ivz + i),
			.e2 = vec(vload)(0, __ie2 + i),
			.m = vec(vload)(0, __im + i),
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

		vec(vstore)(ip.drx, 0, __idrx + i);
		vec(vstore)(ip.dry, 0, __idry + i);
		vec(vstore)(ip.drz, 0, __idrz + i);
		vec(vstore)(ip.ax, 0, __iax + i);
		vec(vstore)(ip.ay, 0, __iay + i);
		vec(vstore)(ip.az, 0, __iaz + i);
		vec(vstore)(ip.m * ip.u, 0, __iu + i);
	}
}


kernel void
nreg_Vkernel(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ivx[],
	global const real_t __ivy[],
	global const real_t __ivz[],
	global const real_t __iax[],
	global const real_t __iay[],
	global const real_t __iaz[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __jvx[],
	global const real_t __jvy[],
	global const real_t __jvz[],
	global const real_t __jax[],
	global const real_t __jay[],
	global const real_t __jaz[],
	const real_t dt,
	global real_t __idvx[],
	global real_t __idvy[],
	global real_t __idvz[],
	global real_t __ik[])
{
	uint_t lid = get_local_id(0);
	uint_t start = get_group_id(0) * get_local_size(0);
	uint_t stride = get_num_groups(0) * get_local_size(0);
	for (uint_t ii = start; ii * SIMD < ni; ii += stride) {
		uint_t i = ii + lid;
		i *= SIMD;
		i = (i+SIMD < ni) ? (i):(ni-SIMD);
		i *= (SIMD < ni);

		vec(Nreg_V_Data) ip = (vec(Nreg_V_Data)){
			.dvx = (real_tn)(0),
			.dvy = (real_tn)(0),
			.dvz = (real_tn)(0),
			.k = (real_tn)(0),
			.vx = vec(vload)(0, __ivx + i),
			.vy = vec(vload)(0, __ivy + i),
			.vz = vec(vload)(0, __ivz + i),
			.ax = vec(vload)(0, __iax + i),
			.ay = vec(vload)(0, __iay + i),
			.az = vec(vload)(0, __iaz + i),
			.m = vec(vload)(0, __im + i),
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

		vec(vstore)(ip.dvx, 0, __idvx + i);
		vec(vstore)(ip.dvy, 0, __idvy + i);
		vec(vstore)(ip.dvz, 0, __idvz + i);
		vec(vstore)(ip.m * ip.k, 0, __ik + i);
	}
}

