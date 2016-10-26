#include "pnacc_kernel_common.h"


kernel void
pnacc_kernel(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
//	const CLIGHT clight,
	constant const CLIGHT * clight,
	global real_t __ipnacc[])
{
	for (uint_t ii = SIMD * get_group_id(0) * get_local_size(0);
				ii < ni;
				ii += SIMD * get_num_groups(0) * get_local_size(0)) {
		uint_t lid = get_local_id(0);
		uint_t i = ii + SIMD * lid;
		i = min(i, ni-SIMD);
		i *= (SIMD < ni);

		PNAcc_Data ip;
		ip.m = vec(vload)(0, __im + i);
		ip.e2 = vec(vload)(0, __ie2 + i);
		ip.rx = vec(vload)(0, &__irdot[(0*NDIM+0)*ni + i]);
		ip.ry = vec(vload)(0, &__irdot[(0*NDIM+1)*ni + i]);
		ip.rz = vec(vload)(0, &__irdot[(0*NDIM+2)*ni + i]);
		ip.vx = vec(vload)(0, &__irdot[(1*NDIM+0)*ni + i]);
		ip.vy = vec(vload)(0, &__irdot[(1*NDIM+1)*ni + i]);
		ip.vz = vec(vload)(0, &__irdot[(1*NDIM+2)*ni + i]);
		ip.pnax = (real_tn)(0);
		ip.pnay = (real_tn)(0);
		ip.pnaz = (real_tn)(0);

		uint_t j = 0;

		#ifdef FAST_LOCAL_MEM
		for (; (j + LSIZE - 1) < nj; j += LSIZE) {
			PNAcc_Data jp;
			jp.m = (real_tn)(__jm[j + lid]);
			jp.e2 = (real_tn)(__je2[j + lid]);
			jp.rx = (real_tn)(__jrdot[(0*NDIM+0)*nj + j + lid]);
			jp.ry = (real_tn)(__jrdot[(0*NDIM+1)*nj + j + lid]);
			jp.rz = (real_tn)(__jrdot[(0*NDIM+2)*nj + j + lid]);
			jp.vx = (real_tn)(__jrdot[(1*NDIM+0)*nj + j + lid]);
			jp.vy = (real_tn)(__jrdot[(1*NDIM+1)*nj + j + lid]);
			jp.vz = (real_tn)(__jrdot[(1*NDIM+2)*nj + j + lid]);
			jp.pnax = (real_tn)(0);
			jp.pnay = (real_tn)(0);
			jp.pnaz = (real_tn)(0);
			barrier(CLK_LOCAL_MEM_FENCE);
			local PNAcc_Data _jp[LSIZE];
			_jp[lid] = jp;
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll
			for (uint_t k = 0; k < LSIZE; ++k) {
				jp = _jp[k];
				ip = pnacc_kernel_core(ip, jp, *clight);
			}
		}
		#endif

		for (; j < nj; ++j) {
			PNAcc_Data jp;
			jp.m = (real_tn)(__jm[j]);
			jp.e2 = (real_tn)(__je2[j]);
			jp.rx = (real_tn)(__jrdot[(0*NDIM+0)*nj + j]);
			jp.ry = (real_tn)(__jrdot[(0*NDIM+1)*nj + j]);
			jp.rz = (real_tn)(__jrdot[(0*NDIM+2)*nj + j]);
			jp.vx = (real_tn)(__jrdot[(1*NDIM+0)*nj + j]);
			jp.vy = (real_tn)(__jrdot[(1*NDIM+1)*nj + j]);
			jp.vz = (real_tn)(__jrdot[(1*NDIM+2)*nj + j]);
			jp.pnax = (real_tn)(0);
			jp.pnay = (real_tn)(0);
			jp.pnaz = (real_tn)(0);
			ip = pnacc_kernel_core(ip, jp, *clight);
		}

		vec(vstore)(ip.pnax, 0, &__ipnacc[(0*NDIM+0)*ni + i]);
		vec(vstore)(ip.pnay, 0, &__ipnacc[(0*NDIM+1)*ni + i]);
		vec(vstore)(ip.pnaz, 0, &__ipnacc[(0*NDIM+2)*ni + i]);
	}
}

