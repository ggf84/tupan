#include "snp_crk_kernel_common.h"


void
snp_crk_kernel_impl(
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
	local Snp_Crk_Data *ip,
	local Snp_Crk_Data *jp)
{
	event_t e[26];
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
		e[8] = async_work_group_copy(ip->_ax, __irdot+(2*NDIM+0)*ni+ii, iN, 0);
		e[9] = async_work_group_copy(ip->_ay, __irdot+(2*NDIM+1)*ni+ii, iN, 0);
		e[10] = async_work_group_copy(ip->_az, __irdot+(2*NDIM+2)*ni+ii, iN, 0);
		e[11] = async_work_group_copy(ip->_jx, __irdot+(3*NDIM+0)*ni+ii, iN, 0);
		e[12] = async_work_group_copy(ip->_jy, __irdot+(3*NDIM+1)*ni+ii, iN, 0);
		e[13] = async_work_group_copy(ip->_jz, __irdot+(3*NDIM+2)*ni+ii, iN, 0);
		e[14] = async_work_group_copy(ip->_Ax, __iadot+(0*NDIM+0)*ni+ii, iN, 0);
		e[15] = async_work_group_copy(ip->_Ay, __iadot+(0*NDIM+1)*ni+ii, iN, 0);
		e[16] = async_work_group_copy(ip->_Az, __iadot+(0*NDIM+2)*ni+ii, iN, 0);
		e[17] = async_work_group_copy(ip->_Jx, __iadot+(1*NDIM+0)*ni+ii, iN, 0);
		e[18] = async_work_group_copy(ip->_Jy, __iadot+(1*NDIM+1)*ni+ii, iN, 0);
		e[19] = async_work_group_copy(ip->_Jz, __iadot+(1*NDIM+2)*ni+ii, iN, 0);
		e[20] = async_work_group_copy(ip->_Sx, __iadot+(2*NDIM+0)*ni+ii, iN, 0);
		e[21] = async_work_group_copy(ip->_Sy, __iadot+(2*NDIM+1)*ni+ii, iN, 0);
		e[22] = async_work_group_copy(ip->_Sz, __iadot+(2*NDIM+2)*ni+ii, iN, 0);
		e[23] = async_work_group_copy(ip->_Cx, __iadot+(3*NDIM+0)*ni+ii, iN, 0);
		e[24] = async_work_group_copy(ip->_Cy, __iadot+(3*NDIM+1)*ni+ii, iN, 0);
		e[25] = async_work_group_copy(ip->_Cz, __iadot+(3*NDIM+2)*ni+ii, iN, 0);
		wait_group_events(26, e);
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
			e[8] = async_work_group_copy(jp->_ax, __jrdot+(2*NDIM+0)*nj+jj, jN, 0);
			e[9] = async_work_group_copy(jp->_ay, __jrdot+(2*NDIM+1)*nj+jj, jN, 0);
			e[10] = async_work_group_copy(jp->_az, __jrdot+(2*NDIM+2)*nj+jj, jN, 0);
			e[11] = async_work_group_copy(jp->_jx, __jrdot+(3*NDIM+0)*nj+jj, jN, 0);
			e[12] = async_work_group_copy(jp->_jy, __jrdot+(3*NDIM+1)*nj+jj, jN, 0);
			e[13] = async_work_group_copy(jp->_jz, __jrdot+(3*NDIM+2)*nj+jj, jN, 0);
			e[14] = async_work_group_copy(jp->_Ax, __jadot+(0*NDIM+0)*nj+jj, jN, 0);
			e[15] = async_work_group_copy(jp->_Ay, __jadot+(0*NDIM+1)*nj+jj, jN, 0);
			e[16] = async_work_group_copy(jp->_Az, __jadot+(0*NDIM+2)*nj+jj, jN, 0);
			e[17] = async_work_group_copy(jp->_Jx, __jadot+(1*NDIM+0)*nj+jj, jN, 0);
			e[18] = async_work_group_copy(jp->_Jy, __jadot+(1*NDIM+1)*nj+jj, jN, 0);
			e[19] = async_work_group_copy(jp->_Jz, __jadot+(1*NDIM+2)*nj+jj, jN, 0);
			e[20] = async_work_group_copy(jp->_Sx, __jadot+(2*NDIM+0)*nj+jj, jN, 0);
			e[21] = async_work_group_copy(jp->_Sy, __jadot+(2*NDIM+1)*nj+jj, jN, 0);
			e[22] = async_work_group_copy(jp->_Sz, __jadot+(2*NDIM+2)*nj+jj, jN, 0);
			e[23] = async_work_group_copy(jp->_Cx, __jadot+(3*NDIM+0)*nj+jj, jN, 0);
			e[24] = async_work_group_copy(jp->_Cy, __jadot+(3*NDIM+1)*nj+jj, jN, 0);
			e[25] = async_work_group_copy(jp->_Cz, __jadot+(3*NDIM+2)*nj+jj, jN, 0);
			wait_group_events(26, e);
			#pragma unroll 64
			for (uint_t j = 0; j < jN; ++j) {
				snp_crk_kernel_core(lid, j, ip, jp);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		e[0] = async_work_group_copy(__iadot+(0*NDIM+0)*ni+ii, ip->_Ax, iN, 0);
		e[1] = async_work_group_copy(__iadot+(0*NDIM+1)*ni+ii, ip->_Ay, iN, 0);
		e[2] = async_work_group_copy(__iadot+(0*NDIM+2)*ni+ii, ip->_Az, iN, 0);
		e[3] = async_work_group_copy(__iadot+(1*NDIM+0)*ni+ii, ip->_Jx, iN, 0);
		e[4] = async_work_group_copy(__iadot+(1*NDIM+1)*ni+ii, ip->_Jy, iN, 0);
		e[5] = async_work_group_copy(__iadot+(1*NDIM+2)*ni+ii, ip->_Jz, iN, 0);
		e[6] = async_work_group_copy(__iadot+(2*NDIM+0)*ni+ii, ip->_Sx, iN, 0);
		e[7] = async_work_group_copy(__iadot+(2*NDIM+1)*ni+ii, ip->_Sy, iN, 0);
		e[8] = async_work_group_copy(__iadot+(2*NDIM+2)*ni+ii, ip->_Sz, iN, 0);
		e[9] = async_work_group_copy(__iadot+(3*NDIM+0)*ni+ii, ip->_Cx, iN, 0);
		e[10] = async_work_group_copy(__iadot+(3*NDIM+1)*ni+ii, ip->_Cy, iN, 0);
		e[11] = async_work_group_copy(__iadot+(3*NDIM+2)*ni+ii, ip->_Cz, iN, 0);
		wait_group_events(12, e);
	}
}


kernel void
snp_crk_kernel_rectangle(
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
	local Snp_Crk_Data _ip;
	local Snp_Crk_Data _jp;

	snp_crk_kernel_impl(
		ni, __im, __ie2, __irdot,
		nj, __jm, __je2, __jrdot,
		__iadot, __jadot,
		&_ip, &_jp
	);

	snp_crk_kernel_impl(
		nj, __jm, __je2, __jrdot,
		ni, __im, __ie2, __irdot,
		__jadot, __iadot,
		&_jp, &_ip
	);
}


kernel void
snp_crk_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	global real_t __iadot[])
{
	local Snp_Crk_Data _ip;
	local Snp_Crk_Data _jp;

	snp_crk_kernel_impl(
		ni, __im, __ie2, __irdot,
		ni, __im, __ie2, __irdot,
		__iadot, __iadot,
		&_ip, &_jp
	);
}

