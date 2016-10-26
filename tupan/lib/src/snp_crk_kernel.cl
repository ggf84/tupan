#include "snp_crk_kernel_common.h"


kernel void
snp_crk_kernel(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	global real_t __iadot[])
{
	for (uint_t ii = SIMD * get_group_id(0) * get_local_size(0);
				ii < ni;
				ii += SIMD * get_num_groups(0) * get_local_size(0)) {
		uint_t lid = get_local_id(0);
		uint_t i = ii + SIMD * lid;
		i = min(i, ni-SIMD);
		i *= (SIMD < ni);

		Snp_Crk_Data ip;
		ip.m = vec(vload)(0, __im + i);
		ip.e2 = vec(vload)(0, __ie2 + i);
		ip.rx = vec(vload)(0, &__irdot[(0*NDIM+0)*ni + i]);
		ip.ry = vec(vload)(0, &__irdot[(0*NDIM+1)*ni + i]);
		ip.rz = vec(vload)(0, &__irdot[(0*NDIM+2)*ni + i]);
		ip.vx = vec(vload)(0, &__irdot[(1*NDIM+0)*ni + i]);
		ip.vy = vec(vload)(0, &__irdot[(1*NDIM+1)*ni + i]);
		ip.vz = vec(vload)(0, &__irdot[(1*NDIM+2)*ni + i]);
		ip.ax = vec(vload)(0, &__irdot[(2*NDIM+0)*ni + i]);
		ip.ay = vec(vload)(0, &__irdot[(2*NDIM+1)*ni + i]);
		ip.az = vec(vload)(0, &__irdot[(2*NDIM+2)*ni + i]);
		ip.jx = vec(vload)(0, &__irdot[(3*NDIM+0)*ni + i]);
		ip.jy = vec(vload)(0, &__irdot[(3*NDIM+1)*ni + i]);
		ip.jz = vec(vload)(0, &__irdot[(3*NDIM+2)*ni + i]);
		ip.Ax = (real_tn)(0);
		ip.Ay = (real_tn)(0);
		ip.Az = (real_tn)(0);
		ip.Jx = (real_tn)(0);
		ip.Jy = (real_tn)(0);
		ip.Jz = (real_tn)(0);
		ip.Sx = (real_tn)(0);
		ip.Sy = (real_tn)(0);
		ip.Sz = (real_tn)(0);
		ip.Cx = (real_tn)(0);
		ip.Cy = (real_tn)(0);
		ip.Cz = (real_tn)(0);

		uint_t j = 0;

		#ifdef FAST_LOCAL_MEM
		for (; (j + LSIZE - 1) < nj; j += LSIZE) {
			Snp_Crk_Data jp;
			jp.m = (real_tn)(__jm[j + lid]);
			jp.e2 = (real_tn)(__je2[j + lid]);
			jp.rx = (real_tn)(__jrdot[(0*NDIM+0)*nj + j + lid]);
			jp.ry = (real_tn)(__jrdot[(0*NDIM+1)*nj + j + lid]);
			jp.rz = (real_tn)(__jrdot[(0*NDIM+2)*nj + j + lid]);
			jp.vx = (real_tn)(__jrdot[(1*NDIM+0)*nj + j + lid]);
			jp.vy = (real_tn)(__jrdot[(1*NDIM+1)*nj + j + lid]);
			jp.vz = (real_tn)(__jrdot[(1*NDIM+2)*nj + j + lid]);
			jp.ax = (real_tn)(__jrdot[(2*NDIM+0)*nj + j + lid]);
			jp.ay = (real_tn)(__jrdot[(2*NDIM+1)*nj + j + lid]);
			jp.az = (real_tn)(__jrdot[(2*NDIM+2)*nj + j + lid]);
			jp.jx = (real_tn)(__jrdot[(3*NDIM+0)*nj + j + lid]);
			jp.jy = (real_tn)(__jrdot[(3*NDIM+1)*nj + j + lid]);
			jp.jz = (real_tn)(__jrdot[(3*NDIM+2)*nj + j + lid]);
			jp.Ax = (real_tn)(0);
			jp.Ay = (real_tn)(0);
			jp.Az = (real_tn)(0);
			jp.Jx = (real_tn)(0);
			jp.Jy = (real_tn)(0);
			jp.Jz = (real_tn)(0);
			jp.Sx = (real_tn)(0);
			jp.Sy = (real_tn)(0);
			jp.Sz = (real_tn)(0);
			jp.Cx = (real_tn)(0);
			jp.Cy = (real_tn)(0);
			jp.Cz = (real_tn)(0);
			barrier(CLK_LOCAL_MEM_FENCE);
			local Snp_Crk_Data _jp[LSIZE];
			_jp[lid] = jp;
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll
			for (uint_t k = 0; k < LSIZE; ++k) {
				jp = _jp[k];
				ip = snp_crk_kernel_core(ip, jp);
			}
		}
		#endif

		for (; j < nj; ++j) {
			Snp_Crk_Data jp;
			jp.m = (real_tn)(__jm[j]);
			jp.e2 = (real_tn)(__je2[j]);
			jp.rx = (real_tn)(__jrdot[(0*NDIM+0)*nj + j]);
			jp.ry = (real_tn)(__jrdot[(0*NDIM+1)*nj + j]);
			jp.rz = (real_tn)(__jrdot[(0*NDIM+2)*nj + j]);
			jp.vx = (real_tn)(__jrdot[(1*NDIM+0)*nj + j]);
			jp.vy = (real_tn)(__jrdot[(1*NDIM+1)*nj + j]);
			jp.vz = (real_tn)(__jrdot[(1*NDIM+2)*nj + j]);
			jp.ax = (real_tn)(__jrdot[(2*NDIM+0)*nj + j]);
			jp.ay = (real_tn)(__jrdot[(2*NDIM+1)*nj + j]);
			jp.az = (real_tn)(__jrdot[(2*NDIM+2)*nj + j]);
			jp.jx = (real_tn)(__jrdot[(3*NDIM+0)*nj + j]);
			jp.jy = (real_tn)(__jrdot[(3*NDIM+1)*nj + j]);
			jp.jz = (real_tn)(__jrdot[(3*NDIM+2)*nj + j]);
			jp.Ax = (real_tn)(0);
			jp.Ay = (real_tn)(0);
			jp.Az = (real_tn)(0);
			jp.Jx = (real_tn)(0);
			jp.Jy = (real_tn)(0);
			jp.Jz = (real_tn)(0);
			jp.Sx = (real_tn)(0);
			jp.Sy = (real_tn)(0);
			jp.Sz = (real_tn)(0);
			jp.Cx = (real_tn)(0);
			jp.Cy = (real_tn)(0);
			jp.Cz = (real_tn)(0);
			ip = snp_crk_kernel_core(ip, jp);
		}

		vec(vstore)(ip.Ax, 0, &__iadot[(0*NDIM+0)*ni + i]);
		vec(vstore)(ip.Ay, 0, &__iadot[(0*NDIM+1)*ni + i]);
		vec(vstore)(ip.Az, 0, &__iadot[(0*NDIM+2)*ni + i]);
		vec(vstore)(ip.Jx, 0, &__iadot[(1*NDIM+0)*ni + i]);
		vec(vstore)(ip.Jy, 0, &__iadot[(1*NDIM+1)*ni + i]);
		vec(vstore)(ip.Jz, 0, &__iadot[(1*NDIM+2)*ni + i]);
		vec(vstore)(ip.Sx, 0, &__iadot[(2*NDIM+0)*ni + i]);
		vec(vstore)(ip.Sy, 0, &__iadot[(2*NDIM+1)*ni + i]);
		vec(vstore)(ip.Sz, 0, &__iadot[(2*NDIM+2)*ni + i]);
		vec(vstore)(ip.Cx, 0, &__iadot[(3*NDIM+0)*ni + i]);
		vec(vstore)(ip.Cy, 0, &__iadot[(3*NDIM+1)*ni + i]);
		vec(vstore)(ip.Cz, 0, &__iadot[(3*NDIM+2)*ni + i]);
	}
}

