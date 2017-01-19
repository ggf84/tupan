#include "sakura_kernel_common.h"


void
sakura_kernel_impl(
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
	global real_t __idrdot[],
	global real_t __jdrdot[],
	local Sakura_Data *ip,
	local Sakura_Data *jp)
{
	event_t e[14];
	uint_t lid = get_local_id(0);
	for (uint_t ii = LSIZE * 1 * get_group_id(0);
				ii < ni;
				ii += LSIZE * 1 * get_num_groups(0)) {
		uint_t iN = min((uint_t)(LSIZE * 1), (ni - ii));
		e[0] = async_work_group_copy(ip->_m, __im+ii, iN, 0);
		e[1] = async_work_group_copy(ip->_e2, __ie2+ii, iN, 0);
		e[2] = async_work_group_copy(ip->_rx, __irdot+(0*NDIM+0)*ni+ii, iN, 0);
		e[3] = async_work_group_copy(ip->_ry, __irdot+(0*NDIM+1)*ni+ii, iN, 0);
		e[4] = async_work_group_copy(ip->_rz, __irdot+(0*NDIM+2)*ni+ii, iN, 0);
		e[5] = async_work_group_copy(ip->_vx, __irdot+(1*NDIM+0)*ni+ii, iN, 0);
		e[6] = async_work_group_copy(ip->_vy, __irdot+(1*NDIM+1)*ni+ii, iN, 0);
		e[7] = async_work_group_copy(ip->_vz, __irdot+(1*NDIM+2)*ni+ii, iN, 0);
		e[8] = async_work_group_copy(ip->_drx, __idrdot+(0*NDIM+0)*ni+ii, iN, 0);
		e[9] = async_work_group_copy(ip->_dry, __idrdot+(0*NDIM+1)*ni+ii, iN, 0);
		e[10] = async_work_group_copy(ip->_drz, __idrdot+(0*NDIM+2)*ni+ii, iN, 0);
		e[11] = async_work_group_copy(ip->_dvx, __idrdot+(1*NDIM+0)*ni+ii, iN, 0);
		e[12] = async_work_group_copy(ip->_dvy, __idrdot+(1*NDIM+1)*ni+ii, iN, 0);
		e[13] = async_work_group_copy(ip->_dvz, __idrdot+(1*NDIM+2)*ni+ii, iN, 0);
		wait_group_events(14, e);
		for (uint_t jj = 0;
					jj < nj;
					jj += LSIZE * 1) {
			uint_t jN = min((uint_t)(LSIZE * 1), (nj - jj));
			e[0] = async_work_group_copy(jp->_m, __jm+jj, jN, 0);
			e[1] = async_work_group_copy(jp->_e2, __je2+jj, jN, 0);
			e[2] = async_work_group_copy(jp->_rx, __jrdot+(0*NDIM+0)*nj+jj, jN, 0);
			e[3] = async_work_group_copy(jp->_ry, __jrdot+(0*NDIM+1)*nj+jj, jN, 0);
			e[4] = async_work_group_copy(jp->_rz, __jrdot+(0*NDIM+2)*nj+jj, jN, 0);
			e[5] = async_work_group_copy(jp->_vx, __jrdot+(1*NDIM+0)*nj+jj, jN, 0);
			e[6] = async_work_group_copy(jp->_vy, __jrdot+(1*NDIM+1)*nj+jj, jN, 0);
			e[7] = async_work_group_copy(jp->_vz, __jrdot+(1*NDIM+2)*nj+jj, jN, 0);
			e[8] = async_work_group_copy(jp->_drx, __jdrdot+(0*NDIM+0)*nj+jj, jN, 0);
			e[9] = async_work_group_copy(jp->_dry, __jdrdot+(0*NDIM+1)*nj+jj, jN, 0);
			e[10] = async_work_group_copy(jp->_drz, __jdrdot+(0*NDIM+2)*nj+jj, jN, 0);
			e[11] = async_work_group_copy(jp->_dvx, __jdrdot+(1*NDIM+0)*nj+jj, jN, 0);
			e[12] = async_work_group_copy(jp->_dvy, __jdrdot+(1*NDIM+1)*nj+jj, jN, 0);
			e[13] = async_work_group_copy(jp->_dvz, __jdrdot+(1*NDIM+2)*nj+jj, jN, 0);
			wait_group_events(14, e);
			#pragma unroll 64
			for (uint_t j = 0; j < jN; ++j) {
				sakura_kernel_core(lid, j, ip, jp, dt, flag);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		e[0] = async_work_group_copy(__idrdot+(0*NDIM+0)*ni+ii, ip->_drx, iN, 0);
		e[1] = async_work_group_copy(__idrdot+(0*NDIM+1)*ni+ii, ip->_dry, iN, 0);
		e[2] = async_work_group_copy(__idrdot+(0*NDIM+2)*ni+ii, ip->_drz, iN, 0);
		e[3] = async_work_group_copy(__idrdot+(1*NDIM+0)*ni+ii, ip->_dvx, iN, 0);
		e[4] = async_work_group_copy(__idrdot+(1*NDIM+1)*ni+ii, ip->_dvy, iN, 0);
		e[5] = async_work_group_copy(__idrdot+(1*NDIM+2)*ni+ii, ip->_dvz, iN, 0);
		wait_group_events(6, e);
	}
}


kernel void
sakura_kernel_rectangle(
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
	global real_t __idrdot[],
	global real_t __jdrdot[])
{
	local Sakura_Data _ip;
	local Sakura_Data _jp;

	sakura_kernel_impl(
		ni, __im, __ie2, __irdot,
		nj, __jm, __je2, __jrdot,
		dt, flag,
		__idrdot, __jdrdot,
		&_ip, &_jp
	);

	sakura_kernel_impl(
		nj, __jm, __je2, __jrdot,
		ni, __im, __ie2, __irdot,
		dt, flag,
		__jdrdot, __idrdot,
		&_jp, &_ip
	);
}


kernel void
sakura_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const real_t dt,
	const int_t flag,
	global real_t __idrdot[])
{
	local Sakura_Data _ip;
	local Sakura_Data _jp;

	sakura_kernel_impl(
		ni, __im, __ie2, __irdot,
		ni, __im, __ie2, __irdot,
		dt, flag,
		__idrdot, __idrdot,
		&_ip, &_jp
	);
}

