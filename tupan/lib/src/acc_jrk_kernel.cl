#include "acc_jrk_kernel_common.h"


void
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
	event_t e[14];
	uint_t lid = get_local_id(0);
	for (uint_t ii = LSIZE * SIMD * get_group_id(0);
				ii < ni;
				ii += LSIZE * SIMD * get_num_groups(0)) {
		uint_t iN = min((uint_t)(LSIZE * SIMD), (ni - ii));
		e[0] = async_work_group_copy(ip->_m, __im+ii, iN, 0);
		e[1] = async_work_group_copy(ip->_e2, __ie2+ii, iN, 0);
		e[2] = async_work_group_copy(ip->_rx, __irdot+(0*NDIM+0)*ni+ii, iN, 0);
		e[3] = async_work_group_copy(ip->_ry, __irdot+(0*NDIM+1)*ni+ii, iN, 0);
		e[4] = async_work_group_copy(ip->_rz, __irdot+(0*NDIM+2)*ni+ii, iN, 0);
		e[5] = async_work_group_copy(ip->_vx, __irdot+(1*NDIM+0)*ni+ii, iN, 0);
		e[6] = async_work_group_copy(ip->_vy, __irdot+(1*NDIM+1)*ni+ii, iN, 0);
		e[7] = async_work_group_copy(ip->_vz, __irdot+(1*NDIM+2)*ni+ii, iN, 0);
		e[8] = async_work_group_copy(ip->_ax, __iadot+(0*NDIM+0)*ni+ii, iN, 0);
		e[9] = async_work_group_copy(ip->_ay, __iadot+(0*NDIM+1)*ni+ii, iN, 0);
		e[10] = async_work_group_copy(ip->_az, __iadot+(0*NDIM+2)*ni+ii, iN, 0);
		e[11] = async_work_group_copy(ip->_jx, __iadot+(1*NDIM+0)*ni+ii, iN, 0);
		e[12] = async_work_group_copy(ip->_jy, __iadot+(1*NDIM+1)*ni+ii, iN, 0);
		e[13] = async_work_group_copy(ip->_jz, __iadot+(1*NDIM+2)*ni+ii, iN, 0);
		wait_group_events(14, e);
		for (uint_t jj = 0;
					jj < nj;
					jj += LSIZE * SIMD) {
			uint_t jN = min((uint_t)(LSIZE * SIMD), (nj - jj));
			e[0] = async_work_group_copy(jp->_m, __jm+jj, jN, 0);
			e[1] = async_work_group_copy(jp->_e2, __je2+jj, jN, 0);
			e[2] = async_work_group_copy(jp->_rx, __jrdot+(0*NDIM+0)*nj+jj, jN, 0);
			e[3] = async_work_group_copy(jp->_ry, __jrdot+(0*NDIM+1)*nj+jj, jN, 0);
			e[4] = async_work_group_copy(jp->_rz, __jrdot+(0*NDIM+2)*nj+jj, jN, 0);
			e[5] = async_work_group_copy(jp->_vx, __jrdot+(1*NDIM+0)*nj+jj, jN, 0);
			e[6] = async_work_group_copy(jp->_vy, __jrdot+(1*NDIM+1)*nj+jj, jN, 0);
			e[7] = async_work_group_copy(jp->_vz, __jrdot+(1*NDIM+2)*nj+jj, jN, 0);
			e[8] = async_work_group_copy(jp->_ax, __jadot+(0*NDIM+0)*nj+jj, jN, 0);
			e[9] = async_work_group_copy(jp->_ay, __jadot+(0*NDIM+1)*nj+jj, jN, 0);
			e[10] = async_work_group_copy(jp->_az, __jadot+(0*NDIM+2)*nj+jj, jN, 0);
			e[11] = async_work_group_copy(jp->_jx, __jadot+(1*NDIM+0)*nj+jj, jN, 0);
			e[12] = async_work_group_copy(jp->_jy, __jadot+(1*NDIM+1)*nj+jj, jN, 0);
			e[13] = async_work_group_copy(jp->_jz, __jadot+(1*NDIM+2)*nj+jj, jN, 0);
			wait_group_events(14, e);
			#pragma unroll 64
			for (uint_t j = 0; j < jN; ++j) {
				acc_jrk_kernel_core(lid, j, ip, jp);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		e[0] = async_work_group_copy(__iadot+(0*NDIM+0)*ni+ii, ip->_ax, iN, 0);
		e[1] = async_work_group_copy(__iadot+(0*NDIM+1)*ni+ii, ip->_ay, iN, 0);
		e[2] = async_work_group_copy(__iadot+(0*NDIM+2)*ni+ii, ip->_az, iN, 0);
		e[3] = async_work_group_copy(__iadot+(1*NDIM+0)*ni+ii, ip->_jx, iN, 0);
		e[4] = async_work_group_copy(__iadot+(1*NDIM+1)*ni+ii, ip->_jy, iN, 0);
		e[5] = async_work_group_copy(__iadot+(1*NDIM+2)*ni+ii, ip->_jz, iN, 0);
		wait_group_events(6, e);
	}
}


kernel void
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


kernel void
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

