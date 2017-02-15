#include "phi_kernel_common.h"


static inline void
phi_kernel_core(
	uint_t ilid,
	uint_t jlid,
	local Phi_Data *ip,
	local Phi_Data *jp)
// flop count: 14
{
	for (uint_t ii = 0; ii < LMSIZE; ii += WGSIZE) {
		uint_t i = ii + ilid;
		real_tn im = ip->m[i];
		real_tn iee = ip->e2[i];
		real_tn irx = ip->rx[i];
		real_tn iry = ip->ry[i];
		real_tn irz = ip->rz[i];
//		real_tn iphi = ip->phi[i];
		for (uint_t jj = 0; jj < LMSIZE; jj += WGSIZE) {
			uint_t j = jj + jlid;
			real_tn ee = iee + jp->e2[j];
			real_tn rx = irx - jp->rx[j];
			real_tn ry = iry - jp->ry[j];
			real_tn rz = irz - jp->rz[j];

			real_tn rr = ee;
			rr += rx * rx + ry * ry + rz * rz;

			real_tn inv_r1 = rsqrt(rr);
			inv_r1 = (rr > ee) ? (inv_r1):(0);

			jp->phi[j] -= im * inv_r1;

//			iphi -= jp->m[j] * inv_r1;
		}
//		ip->phi[i] = iphi;
	}
}


static inline void
phi_kernel_impl(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	global real_t __iphi[],
	global real_t __jphi[],
	local Phi_Data *ip,
	local Phi_Data *jp)
{
	uint_t lid = get_local_id(0);
	for (uint_t ii = LMSIZE * SIMD * get_group_id(0);
				ii < ni;
				ii += LMSIZE * SIMD * get_num_groups(0)) {
		uint_t iN = min((uint_t)(LMSIZE * SIMD), (ni - ii));
		for (uint_t kk = 0; kk < LMSIZE; kk += WGSIZE) {
			uint_t k = kk + lid;
			ip->m[k] = (real_tn)(0);
			ip->e2[k] = (real_tn)(0);
			ip->rx[k] = (real_tn)(0);
			ip->ry[k] = (real_tn)(0);
			ip->rz[k] = (real_tn)(0);
			ip->phi[k] = (real_tn)(0);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		async_work_group_copy(ip->_m, __im+ii, iN, 0);
		async_work_group_copy(ip->_e2, __ie2+ii, iN, 0);
		async_work_group_copy(ip->_rx, __irdot+(0*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_ry, __irdot+(0*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_rz, __irdot+(0*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_phi, __iphi+ii, iN, 0);
		for (uint_t jj = 0;
					jj < nj;
					jj += LMSIZE * SIMD) {
			uint_t jN = min((uint_t)(LMSIZE * SIMD), (nj - jj));
			for (uint_t kk = 0; kk < LMSIZE; kk += WGSIZE) {
				uint_t k = kk + lid;
				jp->m[k] = (real_tn)(0);
				jp->e2[k] = (real_tn)(0);
				jp->rx[k] = (real_tn)(0);
				jp->ry[k] = (real_tn)(0);
				jp->rz[k] = (real_tn)(0);
				jp->phi[k] = (real_tn)(0);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			async_work_group_copy(jp->_m, __jm+jj, jN, 0);
			async_work_group_copy(jp->_e2, __je2+jj, jN, 0);
			async_work_group_copy(jp->_rx, __jrdot+(0*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_ry, __jrdot+(0*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_rz, __jrdot+(0*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_phi, __jphi+jj, jN, 0);
			barrier(CLK_LOCAL_MEM_FENCE);
			for (uint_t k = 0; k < SIMD; ++k) {
//				#pragma unroll
				for (uint_t l = 0; l < WGSIZE; ++l) {
					phi_kernel_core(l, lid, jp, ip);
//					phi_kernel_core((lid + l) % WGSIZE, lid, jp, ip);
				}
				for (uint_t kk = 0; kk < LMSIZE; kk += WGSIZE) {
					uint_t i = kk + lid;
					shuff(ip->m[i], SIMD);
					shuff(ip->e2[i], SIMD);
					shuff(ip->rx[i], SIMD);
					shuff(ip->ry[i], SIMD);
					shuff(ip->rz[i], SIMD);
					shuff(ip->phi[i], SIMD);
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		async_work_group_copy(__iphi+ii, ip->_phi, iN, 0);
	}
}


kernel __attribute__((reqd_work_group_size(WGSIZE, 1, 1))) void
phi_kernel_rectangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	global real_t __iphi[],
	global real_t __jphi[])
{
	local Phi_Data _ip;
	local Phi_Data _jp;

	phi_kernel_impl(
		ni, __im, __ie2, __irdot,
		nj, __jm, __je2, __jrdot,
		__iphi, __jphi,
		&_ip, &_jp
	);

	phi_kernel_impl(
		nj, __jm, __je2, __jrdot,
		ni, __im, __ie2, __irdot,
		__jphi, __iphi,
		&_jp, &_ip
	);
}


kernel __attribute__((reqd_work_group_size(WGSIZE, 1, 1))) void
phi_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	global real_t __iphi[])
{
	local Phi_Data _ip;
	local Phi_Data _jp;

	phi_kernel_impl(
		ni, __im, __ie2, __irdot,
		ni, __im, __ie2, __irdot,
		__iphi, __iphi,
		&_ip, &_jp
	);
}

