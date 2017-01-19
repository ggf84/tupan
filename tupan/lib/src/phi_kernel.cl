#include "phi_kernel_common.h"


void
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
	event_t e;
	for (uint_t ii = LSIZE * SIMD * get_group_id(0);
				ii < ni;
				ii += LSIZE * SIMD * get_num_groups(0)) {
		uint_t iN = min((uint_t)(LSIZE * SIMD), (ni - ii));
		e = async_work_group_copy(ip->_m, __im+ii, iN, 0);
		e = async_work_group_copy(ip->_e2, __ie2+ii, iN, 0);
		e = async_work_group_copy(ip->_rx, __irdot+(0*NDIM+0)*ni+ii, iN, 0);
		e = async_work_group_copy(ip->_ry, __irdot+(0*NDIM+1)*ni+ii, iN, 0);
		e = async_work_group_copy(ip->_rz, __irdot+(0*NDIM+2)*ni+ii, iN, 0);
		e = async_work_group_copy(ip->_phi, __iphi+ii, iN, 0);
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
			e = async_work_group_copy(jp->_phi, __jphi+jj, jN, 0);
			barrier(CLK_LOCAL_MEM_FENCE);
			for (uint_t i = get_local_id(0);
						i < LSIZE;
						i += get_local_size(0)) {
				#pragma unroll 32
				for (uint_t j = 0; j < jN; ++j) {
					phi_kernel_core(i, j, ip, jp);
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		e = async_work_group_copy(__iphi+ii, ip->_phi, iN, 0);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


kernel void
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


kernel void
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

