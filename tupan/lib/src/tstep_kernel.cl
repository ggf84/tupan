#include "tstep_kernel_common.h"


void
tstep_kernel_impl(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	const real_t eta,
	global real_t __idt_a[],
	global real_t __idt_b[],
	global real_t __jdt_a[],
	global real_t __jdt_b[],
	local Tstep_Data *ip,
	local Tstep_Data *jp)
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
		e = async_work_group_copy(ip->_vx, __irdot+(1*NDIM+0)*ni+ii, iN, 0);
		e = async_work_group_copy(ip->_vy, __irdot+(1*NDIM+1)*ni+ii, iN, 0);
		e = async_work_group_copy(ip->_vz, __irdot+(1*NDIM+2)*ni+ii, iN, 0);
		e = async_work_group_copy(ip->_w2_a, __idt_a+ii, iN, 0);
		e = async_work_group_copy(ip->_w2_b, __idt_b+ii, iN, 0);
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
			e = async_work_group_copy(jp->_w2_a, __jdt_a+jj, jN, 0);
			e = async_work_group_copy(jp->_w2_b, __jdt_b+jj, jN, 0);
			barrier(CLK_LOCAL_MEM_FENCE);
			for (uint_t i = get_local_id(0);
						i < LSIZE;
						i += get_local_size(0)) {
				#pragma unroll 32
				for (uint_t j = 0; j < jN; ++j) {
					tstep_kernel_core(i, j, ip, jp, eta);
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		for (uint_t i = get_local_id(0);
					i < LSIZE;
					i += get_local_size(0)) {
			ip->w2_a[i] = eta * rsqrt(ip->w2_a[i]);
			ip->w2_b[i] = eta * rsqrt(ip->w2_b[i]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		e = async_work_group_copy(__idt_a+ii, ip->_w2_a, iN, 0);
		e = async_work_group_copy(__idt_b+ii, ip->_w2_b, iN, 0);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


kernel void
tstep_kernel_rectangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	const real_t eta,
	global real_t __idt_a[],
	global real_t __idt_b[],
	global real_t __jdt_a[],
	global real_t __jdt_b[])
{
	local Tstep_Data _ip;
	local Tstep_Data _jp;

	tstep_kernel_impl(
		ni, __im, __ie2, __irdot,
		nj, __jm, __je2, __jrdot,
		eta,
		__idt_a, __idt_b,
		__jdt_a, __jdt_b,
		&_ip, &_jp
	);

	tstep_kernel_impl(
		nj, __jm, __je2, __jrdot,
		ni, __im, __ie2, __irdot,
		eta,
		__jdt_a, __jdt_b,
		__idt_a, __idt_b,
		&_jp, &_ip
	);
}


kernel void
tstep_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const real_t eta,
	global real_t __idt_a[],
	global real_t __idt_b[])
{
	local Tstep_Data _ip;
	local Tstep_Data _jp;

	tstep_kernel_impl(
		ni, __im, __ie2, __irdot,
		ni, __im, __ie2, __irdot,
		eta,
		__idt_a, __idt_b,
		__idt_a, __idt_b,
		&_ip, &_jp
	);
}

