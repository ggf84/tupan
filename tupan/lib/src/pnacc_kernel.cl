#include "pnacc_kernel_common.h"


void
pnacc_kernel_impl(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	const CLIGHT clight,
	global real_t __ipnacc[],
	global real_t __jpnacc[],
	local PNAcc_Data *ip,
	local PNAcc_Data *jp)
{
	event_t e;
	uint_t lid = get_local_id(0);
	for (uint_t ii = LSIZE * SIMD * get_group_id(0);
				ii < ni;
				ii += LSIZE * SIMD * get_num_groups(0)) {
		uint_t iN = min((uint_t)(LSIZE * SIMD), (ni - ii));
		e = async_work_group_copy(ip->_m, __im+ii, iN, 0);
		e = async_work_group_copy(ip->_e2, __ie2+ii, iN, 0);
		e = async_work_group_copy(ip->_rx, __irdot+(0*NDIM+0)*ni+ii, iN, 0);
		e = async_work_group_copy(ip->_ry, __irdot+(0*NDIM+1)*ni+ii, iN, 0);
		e = async_work_group_copy(ip->_rz, __irdot+(0*NDIM+2)*ni+ii, iN, 0);
		e = async_work_group_copy(ip->_vx, __irdot+(1*NDIM+0)*ni+ii, iN, 0);
		e = async_work_group_copy(ip->_vy, __irdot+(1*NDIM+1)*ni+ii, iN, 0);
		e = async_work_group_copy(ip->_vz, __irdot+(1*NDIM+2)*ni+ii, iN, 0);
		e = async_work_group_copy(ip->_pnax, __ipnacc+(0*NDIM+0)*ni+ii, iN, 0);
		e = async_work_group_copy(ip->_pnay, __ipnacc+(0*NDIM+1)*ni+ii, iN, 0);
		e = async_work_group_copy(ip->_pnaz, __ipnacc+(0*NDIM+2)*ni+ii, iN, 0);
		barrier(CLK_LOCAL_MEM_FENCE);
		for (uint_t jj = 0;
					jj < nj;
					jj += LSIZE * SIMD) {
			uint_t jN = min((uint_t)(LSIZE * SIMD), (nj - jj));
			e = async_work_group_copy(jp->_m, __jm+jj, jN, 0);
			e = async_work_group_copy(jp->_e2, __je2+jj, jN, 0);
			e = async_work_group_copy(jp->_rx, __jrdot+(0*NDIM+0)*nj+jj, jN, 0);
			e = async_work_group_copy(jp->_ry, __jrdot+(0*NDIM+1)*nj+jj, jN, 0);
			e = async_work_group_copy(jp->_rz, __jrdot+(0*NDIM+2)*nj+jj, jN, 0);
			e = async_work_group_copy(jp->_vx, __jrdot+(1*NDIM+0)*nj+jj, jN, 0);
			e = async_work_group_copy(jp->_vy, __jrdot+(1*NDIM+1)*nj+jj, jN, 0);
			e = async_work_group_copy(jp->_vz, __jrdot+(1*NDIM+2)*nj+jj, jN, 0);
			e = async_work_group_copy(jp->_pnax, __jpnacc+(0*NDIM+0)*nj+jj, jN, 0);
			e = async_work_group_copy(jp->_pnay, __jpnacc+(0*NDIM+1)*nj+jj, jN, 0);
			e = async_work_group_copy(jp->_pnaz, __jpnacc+(0*NDIM+2)*nj+jj, jN, 0);
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll 32
			for (uint_t j = 0; j < jN; ++j) {
				pnacc_kernel_core(lid, j, ip, jp, clight);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		e = async_work_group_copy(__ipnacc+(0*NDIM+0)*ni+ii, ip->_pnax, iN, 0);
		e = async_work_group_copy(__ipnacc+(0*NDIM+1)*ni+ii, ip->_pnay, iN, 0);
		e = async_work_group_copy(__ipnacc+(0*NDIM+2)*ni+ii, ip->_pnaz, iN, 0);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


kernel void
pnacc_kernel_rectangle(
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
	global real_t __ipnacc[],
	global real_t __jpnacc[])
{
	local PNAcc_Data _ip;
	local PNAcc_Data _jp;

	pnacc_kernel_impl(
		ni, __im, __ie2, __irdot,
		nj, __jm, __je2, __jrdot,
		*clight,
		__ipnacc, __jpnacc,
		&_ip, &_jp
	);

	pnacc_kernel_impl(
		nj, __jm, __je2, __jrdot,
		ni, __im, __ie2, __irdot,
		*clight,
		__jpnacc, __ipnacc,
		&_jp, &_ip
	);
}


kernel void
pnacc_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
//	const CLIGHT clight,
	constant const CLIGHT * clight,
	global real_t __ipnacc[])
{
	local PNAcc_Data _ip;
	local PNAcc_Data _jp;

	pnacc_kernel_impl(
		ni, __im, __ie2, __irdot,
		ni, __im, __ie2, __irdot,
		*clight,
		__ipnacc, __ipnacc,
		&_ip, &_jp
	);
}

