#include "acc_kernel_common.h"


static inline void
acc_kernel_core(
	uint_t iN, uint_t jN,
	local Acc_Data *ip,
	local Acc_Data *jp)
// flop count: 21
{
	for (uint_t i = get_local_id(0);
				i < iN;
				i += get_local_size(0)) {
		real_tn iee = ip->e2[i];
		real_tn irx = ip->rx[i];
		real_tn iry = ip->ry[i];
		real_tn irz = ip->rz[i];
		real_tn iax = ip->ax[i];
		real_tn iay = ip->ay[i];
		real_tn iaz = ip->az[i];
		#pragma unroll 32
		for (uint_t j = 0; j < jN; ++j) {
			real_tn ee = iee + jp->_e2[j];
			real_tn rx = irx - jp->_rx[j];
			real_tn ry = iry - jp->_ry[j];
			real_tn rz = irz - jp->_rz[j];

			real_tn rr = ee;
			rr += rx * rx + ry * ry + rz * rz;

			real_tn inv_r3 = rsqrt(rr);
			inv_r3 = (rr > ee) ? (inv_r3):(0);
			inv_r3 *= inv_r3 * inv_r3;

			real_tn jm_r3 = jp->_m[j] * inv_r3;

			iax -= jm_r3 * rx;
			iay -= jm_r3 * ry;
			iaz -= jm_r3 * rz;
		}
		ip->ax[i] = iax;
		ip->ay[i] = iay;
		ip->az[i] = iaz;
	}
}


static inline void
acc_kernel_impl(
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
	local Acc_Data *ip,
	local Acc_Data *jp)
{
	for (uint_t ii = LSIZE * SIMD * get_group_id(0);
				ii < ni;
				ii += LSIZE * SIMD * get_num_groups(0)) {
		uint_t iN = min((uint_t)(LSIZE * SIMD), (ni - ii));
		async_work_group_copy(ip->_m, __im+ii, iN, 0);
		async_work_group_copy(ip->_e2, __ie2+ii, iN, 0);
		async_work_group_copy(ip->_rx, __irdot+(0*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_ry, __irdot+(0*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_rz, __irdot+(0*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_ax, __iadot+(0*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_ay, __iadot+(0*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_az, __iadot+(0*NDIM+2)*ni+ii, iN, 0);
		for (uint_t jj = 0;
					jj < nj;
					jj += LSIZE * SIMD) {
			uint_t jN = min((uint_t)(LSIZE * SIMD), (nj - jj));
			async_work_group_copy(jp->_m, __jm+jj, jN, 0);
			async_work_group_copy(jp->_e2, __je2+jj, jN, 0);
			async_work_group_copy(jp->_rx, __jrdot+(0*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_ry, __jrdot+(0*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_rz, __jrdot+(0*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_ax, __jadot+(0*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_ay, __jadot+(0*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_az, __jadot+(0*NDIM+2)*nj+jj, jN, 0);
			barrier(CLK_LOCAL_MEM_FENCE);
			acc_kernel_core(LSIZE, jN, ip, jp);
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		async_work_group_copy(__iadot+(0*NDIM+0)*ni+ii, ip->_ax, iN, 0);
		async_work_group_copy(__iadot+(0*NDIM+1)*ni+ii, ip->_ay, iN, 0);
		async_work_group_copy(__iadot+(0*NDIM+2)*ni+ii, ip->_az, iN, 0);
	}
}


kernel void
acc_kernel_rectangle(
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
	local Acc_Data _ip;
	local Acc_Data _jp;

	acc_kernel_impl(
		ni, __im, __ie2, __irdot,
		nj, __jm, __je2, __jrdot,
		__iadot, __jadot,
		&_ip, &_jp
	);

	acc_kernel_impl(
		nj, __jm, __je2, __jrdot,
		ni, __im, __ie2, __irdot,
		__jadot, __iadot,
		&_jp, &_ip
	);
}


kernel void
acc_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	global real_t __iadot[])
{
	local Acc_Data _ip;
	local Acc_Data _jp;

	acc_kernel_impl(
		ni, __im, __ie2, __irdot,
		ni, __im, __ie2, __irdot,
		__iadot, __iadot,
		&_ip, &_jp
	);
}

