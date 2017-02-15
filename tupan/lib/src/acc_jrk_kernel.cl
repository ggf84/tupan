#include "acc_jrk_kernel_common.h"


static inline void
acc_jrk_kernel_core(
	uint_t ilid,
	uint_t jlid,
	local Acc_Jrk_Data *ip,
	local Acc_Jrk_Data *jp)
// flop count: 43
{
	for (uint_t ii = 0; ii < LMSIZE; ii += WGSIZE) {
		uint_t i = ii + ilid;
		real_tn im = ip->m[i];
		real_tn iee = ip->e2[i];
		real_tn irx = ip->rx[i];
		real_tn iry = ip->ry[i];
		real_tn irz = ip->rz[i];
		real_tn ivx = ip->vx[i];
		real_tn ivy = ip->vy[i];
		real_tn ivz = ip->vz[i];
//		real_tn iax = ip->ax[i];
//		real_tn iay = ip->ay[i];
//		real_tn iaz = ip->az[i];
//		real_tn ijx = ip->jx[i];
//		real_tn ijy = ip->jy[i];
//		real_tn ijz = ip->jz[i];
		for (uint_t jj = 0; jj < LMSIZE; jj += WGSIZE) {
			uint_t j = jj + jlid;
			real_tn ee = iee + jp->e2[j];
			real_tn rx = irx - jp->rx[j];
			real_tn ry = iry - jp->ry[j];
			real_tn rz = irz - jp->rz[j];
			real_tn vx = ivx - jp->vx[j];
			real_tn vy = ivy - jp->vy[j];
			real_tn vz = ivz - jp->vz[j];

			real_tn rr = ee;
			rr += rx * rx + ry * ry + rz * rz;

			real_tn inv_r3 = rsqrt(rr);
			inv_r3 = (rr > ee) ? (inv_r3):(0);
			real_tn inv_r2 = inv_r3 * inv_r3;
			inv_r3 *= inv_r2;
			inv_r2 *= -3;

			real_tn s1 = rx * vx + ry * vy + rz * vz;

			real_tn q1 = inv_r2 * (s1);
			vx += q1 * rx;
			vy += q1 * ry;
			vz += q1 * rz;

			real_tn im_r3 = im * inv_r3;
			jp->ax[j] += im_r3 * rx;
			jp->ay[j] += im_r3 * ry;
			jp->az[j] += im_r3 * rz;
			jp->jx[j] += im_r3 * vx;
			jp->jy[j] += im_r3 * vy;
			jp->jz[j] += im_r3 * vz;

//			real_tn jm_r3 = jp->m[j] * inv_r3;
//			iax -= jm_r3 * rx;
//			iay -= jm_r3 * ry;
//			iaz -= jm_r3 * rz;
//			ijx -= jm_r3 * vx;
//			ijy -= jm_r3 * vy;
//			ijz -= jm_r3 * vz;
		}
//		ip->ax[i] = iax;
//		ip->ay[i] = iay;
//		ip->az[i] = iaz;
//		ip->jx[i] = ijx;
//		ip->jy[i] = ijy;
//		ip->jz[i] = ijz;
	}
}


static inline void
acc_jrk_kernel_impl(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	global real_t __iadot[],
	global real_t __jadot[],
	local Acc_Jrk_Data *ip,
	local Acc_Jrk_Data *jp)
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
			ip->vx[k] = (real_tn)(0);
			ip->vy[k] = (real_tn)(0);
			ip->vz[k] = (real_tn)(0);
			ip->ax[k] = (real_tn)(0);
			ip->ay[k] = (real_tn)(0);
			ip->az[k] = (real_tn)(0);
			ip->jx[k] = (real_tn)(0);
			ip->jy[k] = (real_tn)(0);
			ip->jz[k] = (real_tn)(0);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		async_work_group_copy(ip->_m, __im+ii, iN, 0);
		async_work_group_copy(ip->_e2, __ie2+ii, iN, 0);
		async_work_group_copy(ip->_rx, __irdot+(0*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_ry, __irdot+(0*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_rz, __irdot+(0*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_vx, __irdot+(1*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_vy, __irdot+(1*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_vz, __irdot+(1*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_ax, __iadot+(0*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_ay, __iadot+(0*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_az, __iadot+(0*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_jx, __iadot+(1*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_jy, __iadot+(1*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_jz, __iadot+(1*NDIM+2)*ni+ii, iN, 0);
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
				jp->vx[k] = (real_tn)(0);
				jp->vy[k] = (real_tn)(0);
				jp->vz[k] = (real_tn)(0);
				jp->ax[k] = (real_tn)(0);
				jp->ay[k] = (real_tn)(0);
				jp->az[k] = (real_tn)(0);
				jp->jx[k] = (real_tn)(0);
				jp->jy[k] = (real_tn)(0);
				jp->jz[k] = (real_tn)(0);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			async_work_group_copy(jp->_m, __jm+jj, jN, 0);
			async_work_group_copy(jp->_e2, __je2+jj, jN, 0);
			async_work_group_copy(jp->_rx, __jrdot+(0*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_ry, __jrdot+(0*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_rz, __jrdot+(0*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_vx, __jrdot+(1*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_vy, __jrdot+(1*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_vz, __jrdot+(1*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_ax, __jadot+(0*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_ay, __jadot+(0*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_az, __jadot+(0*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_jx, __jadot+(1*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_jy, __jadot+(1*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_jz, __jadot+(1*NDIM+2)*nj+jj, jN, 0);
			barrier(CLK_LOCAL_MEM_FENCE);
			for (uint_t k = 0; k < SIMD; ++k) {
//				#pragma unroll
				for (uint_t l = 0; l < WGSIZE; ++l) {
					acc_jrk_kernel_core(l, lid, jp, ip);
//					acc_jrk_kernel_core((lid + l) % WGSIZE, lid, jp, ip);
				}
				for (uint_t kk = 0; kk < LMSIZE; kk += WGSIZE) {
					uint_t i = kk + lid;
					shuff(ip->m[i], SIMD);
					shuff(ip->e2[i], SIMD);
					shuff(ip->rx[i], SIMD);
					shuff(ip->ry[i], SIMD);
					shuff(ip->rz[i], SIMD);
					shuff(ip->vx[i], SIMD);
					shuff(ip->vy[i], SIMD);
					shuff(ip->vz[i], SIMD);
					shuff(ip->ax[i], SIMD);
					shuff(ip->ay[i], SIMD);
					shuff(ip->az[i], SIMD);
					shuff(ip->jx[i], SIMD);
					shuff(ip->jy[i], SIMD);
					shuff(ip->jz[i], SIMD);
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		async_work_group_copy(__iadot+(0*NDIM+0)*ni+ii, ip->_ax, iN, 0);
		async_work_group_copy(__iadot+(0*NDIM+1)*ni+ii, ip->_ay, iN, 0);
		async_work_group_copy(__iadot+(0*NDIM+2)*ni+ii, ip->_az, iN, 0);
		async_work_group_copy(__iadot+(1*NDIM+0)*ni+ii, ip->_jx, iN, 0);
		async_work_group_copy(__iadot+(1*NDIM+1)*ni+ii, ip->_jy, iN, 0);
		async_work_group_copy(__iadot+(1*NDIM+2)*ni+ii, ip->_jz, iN, 0);
	}
}


kernel __attribute__((reqd_work_group_size(WGSIZE, 1, 1))) void
acc_jrk_kernel_rectangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	global real_t __iadot[],
	global real_t __jadot[])
{
	local Acc_Jrk_Data _ip;
	local Acc_Jrk_Data _jp;

	acc_jrk_kernel_impl(
		ni, __im, __ie2, __irdot,
		nj, __jm, __je2, __jrdot,
		__iadot, __jadot,
		&_ip, &_jp
	);

	acc_jrk_kernel_impl(
		nj, __jm, __je2, __jrdot,
		ni, __im, __ie2, __irdot,
		__jadot, __iadot,
		&_jp, &_ip
	);
}


kernel __attribute__((reqd_work_group_size(WGSIZE, 1, 1))) void
acc_jrk_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	global real_t __iadot[])
{
	local Acc_Jrk_Data _ip;
	local Acc_Jrk_Data _jp;

	acc_jrk_kernel_impl(
		ni, __im, __ie2, __irdot,
		ni, __im, __ie2, __irdot,
		__iadot, __iadot,
		&_ip, &_jp
	);
}

