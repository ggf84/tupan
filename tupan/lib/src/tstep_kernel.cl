#include "tstep_kernel_common.h"


static inline void
tstep_kernel_core(
	local Tstep_Data *ip,
	local Tstep_Data *jp,
	const real_t eta)
// flop count: 42
{
	uint_t ilid = get_local_id(0);
//	#pragma unroll 8
	for (uint_t l = 0; l < WGSIZE; ++l) {
		uint_t jlid = l;//(ilid + l) % WGSIZE;
		#pragma unroll
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
			real_tn iw2_a = ip->w2_a[i];
			real_tn iw2_b = ip->w2_b[i];
//			#pragma unroll
			for (uint_t k = 0; k < SIMD; ++k) {
				#pragma unroll
				for (uint_t jj = 0; jj < LMSIZE; jj += WGSIZE) {
					uint_t j = jj + jlid;
					real_tn m_r3 = im + jp->m[j];
					real_tn ee = iee + jp->e2[j];
					real_tn rx = irx - jp->rx[j];
					real_tn ry = iry - jp->ry[j];
					real_tn rz = irz - jp->rz[j];
					real_tn vx = ivx - jp->vx[j];
					real_tn vy = ivy - jp->vy[j];
					real_tn vz = ivz - jp->vz[j];

					real_tn rr = ee;
					rr        += rx * rx + ry * ry + rz * rz;
					real_tn rv = rx * vx + ry * vy + rz * vz;
					real_tn vv = vx * vx + vy * vy + vz * vz;

					real_tn inv_r2 = rsqrt(rr);
					m_r3 *= inv_r2;
					inv_r2 *= inv_r2;
					m_r3 *= 2 * inv_r2;

					real_tn m_r5 = m_r3 * inv_r2;
					m_r3 += vv * inv_r2;
					rv *= eta * rsqrt(m_r3);
					m_r5 += m_r3 * inv_r2;
					m_r3 -= m_r5 * rv;

					m_r3 = (rr > ee && jp->m[j] != 0) ? (m_r3):(0);

					iw2_a = fmax(m_r3, iw2_a);
					iw2_b += m_r3;
				}
				shuff(im, SIMD);
				shuff(iee, SIMD);
				shuff(irx, SIMD);
				shuff(iry, SIMD);
				shuff(irz, SIMD);
				shuff(ivx, SIMD);
				shuff(ivy, SIMD);
				shuff(ivz, SIMD);
				shuff(iw2_a, SIMD);
				shuff(iw2_b, SIMD);
			}
			ip->w2_a[i] = iw2_a;
			ip->w2_b[i] = iw2_b;
		}
	}
}


static inline void
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
	for (uint_t ii = LMSIZE * SIMD * get_group_id(0);
				ii < ni;
				ii += LMSIZE * SIMD * get_num_groups(0)) {
		uint_t iN = min((uint_t)(LMSIZE * SIMD), (ni - ii));
		for (uint_t kk = 0; kk < LMSIZE; kk += WGSIZE) {
			uint_t k = kk + get_local_id(0);
			ip->m[k] = (real_tn)(0);
			ip->e2[k] = (real_tn)(0);
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
		async_work_group_copy(ip->_w2_a, __idt_a+ii, iN, 0);
		async_work_group_copy(ip->_w2_b, __idt_b+ii, iN, 0);
		for (uint_t jj = 0;
					jj < nj;
					jj += LMSIZE * SIMD) {
			uint_t jN = min((uint_t)(LMSIZE * SIMD), (nj - jj));
			for (uint_t kk = 0; kk < LMSIZE; kk += WGSIZE) {
				uint_t k = kk + get_local_id(0);
				jp->m[k] = (real_tn)(0);
				jp->e2[k] = (real_tn)(0);
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
			async_work_group_copy(jp->_w2_a, __jdt_a+jj, jN, 0);
			async_work_group_copy(jp->_w2_b, __jdt_b+jj, jN, 0);
			barrier(CLK_LOCAL_MEM_FENCE);
			tstep_kernel_core(ip, jp, eta);
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		for (uint_t kk = 0; kk < LMSIZE; kk += WGSIZE) {
			uint_t k = kk + get_local_id(0);
			ip->w2_a[k] = eta * rsqrt(ip->w2_a[k]);
			ip->w2_b[k] = eta * rsqrt(ip->w2_b[k]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		async_work_group_copy(__idt_a+ii, ip->_w2_a, iN, 0);
		async_work_group_copy(__idt_b+ii, ip->_w2_b, iN, 0);
	}
}


kernel __attribute__((reqd_work_group_size(WGSIZE, 1, 1))) void
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


kernel __attribute__((reqd_work_group_size(WGSIZE, 1, 1))) void
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

