#include "sakura_kernel_common.h"


kernel void
sakura_kernel(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	const real_t dt,
	const int_t flag,
	global real_t __idrdot[])
{
	for (uint_t ii = 1 * get_group_id(0) * get_local_size(0);
				ii < ni;
				ii += 1 * get_num_groups(0) * get_local_size(0)) {
		uint_t lid = get_local_id(0);
		uint_t i = ii + 1 * lid;
		i = min(i, ni-1);
		i *= (1 < ni);

		Sakura_Data ip;
		ip.m = vload1(0, __im + i);
		ip.e2 = vload1(0, __ie2 + i);
		ip.rx = vload1(0, &__irdot[(0*NDIM+0)*ni + i]);
		ip.ry = vload1(0, &__irdot[(0*NDIM+1)*ni + i]);
		ip.rz = vload1(0, &__irdot[(0*NDIM+2)*ni + i]);
		ip.vx = vload1(0, &__irdot[(1*NDIM+0)*ni + i]);
		ip.vy = vload1(0, &__irdot[(1*NDIM+1)*ni + i]);
		ip.vz = vload1(0, &__irdot[(1*NDIM+2)*ni + i]);
		ip.drx = (real_t1)(0);
		ip.dry = (real_t1)(0);
		ip.drz = (real_t1)(0);
		ip.dvx = (real_t1)(0);
		ip.dvy = (real_t1)(0);
		ip.dvz = (real_t1)(0);

		uint_t j = 0;

		#ifdef FAST_LOCAL_MEM
		for (; (j + LSIZE - 1) < nj; j += LSIZE) {
			Sakura_Data jp;
			jp.m = (real_t1)(__jm[j + lid]);
			jp.e2 = (real_t1)(__je2[j + lid]);
			jp.rx = (real_t1)(__jrdot[(0*NDIM+0)*nj + j + lid]);
			jp.ry = (real_t1)(__jrdot[(0*NDIM+1)*nj + j + lid]);
			jp.rz = (real_t1)(__jrdot[(0*NDIM+2)*nj + j + lid]);
			jp.vx = (real_t1)(__jrdot[(1*NDIM+0)*nj + j + lid]);
			jp.vy = (real_t1)(__jrdot[(1*NDIM+1)*nj + j + lid]);
			jp.vz = (real_t1)(__jrdot[(1*NDIM+2)*nj + j + lid]);
			jp.drx = (real_t1)(0);
			jp.dry = (real_t1)(0);
			jp.drz = (real_t1)(0);
			jp.dvx = (real_t1)(0);
			jp.dvy = (real_t1)(0);
			jp.dvz = (real_t1)(0);
			barrier(CLK_LOCAL_MEM_FENCE);
			local Sakura_Data _jp[LSIZE];
			_jp[lid] = jp;
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll
			for (uint_t k = 0; k < LSIZE; ++k) {
				jp = _jp[k];
				ip = sakura_kernel_core(ip, jp, dt, flag);
			}
		}
		#endif

		for (; j < nj; ++j) {
			Sakura_Data jp;
			jp.m = (real_t1)(__jm[j]);
			jp.e2 = (real_t1)(__je2[j]);
			jp.rx = (real_t1)(__jrdot[(0*NDIM+0)*nj + j]);
			jp.ry = (real_t1)(__jrdot[(0*NDIM+1)*nj + j]);
			jp.rz = (real_t1)(__jrdot[(0*NDIM+2)*nj + j]);
			jp.vx = (real_t1)(__jrdot[(1*NDIM+0)*nj + j]);
			jp.vy = (real_t1)(__jrdot[(1*NDIM+1)*nj + j]);
			jp.vz = (real_t1)(__jrdot[(1*NDIM+2)*nj + j]);
			jp.drx = (real_t1)(0);
			jp.dry = (real_t1)(0);
			jp.drz = (real_t1)(0);
			jp.dvx = (real_t1)(0);
			jp.dvy = (real_t1)(0);
			jp.dvz = (real_t1)(0);
			ip = sakura_kernel_core(ip, jp, dt, flag);
		}

		vstore1(ip.drx, 0, &__idrdot[(0*NDIM+0)*ni + i]);
		vstore1(ip.dry, 0, &__idrdot[(0*NDIM+1)*ni + i]);
		vstore1(ip.drz, 0, &__idrdot[(0*NDIM+2)*ni + i]);
		vstore1(ip.dvx, 0, &__idrdot[(1*NDIM+0)*ni + i]);
		vstore1(ip.dvy, 0, &__idrdot[(1*NDIM+1)*ni + i]);
		vstore1(ip.dvz, 0, &__idrdot[(1*NDIM+2)*ni + i]);
	}
}

