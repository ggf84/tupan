#include "pnacc_kernel_common.h"


static inline void
pnacc_kernel_core(
	local PNAcc_Data *ip,
	local PNAcc_Data *jp,
	const CLIGHT clight)
// flop count: 36+???
{
	for (uint_t i = get_local_id(0);
				i < LSIZE;
				i += get_local_size(0)) {
		real_tn im = ip->m[i];
		real_tn iee = ip->e2[i];
		real_tn irx = ip->rx[i];
		real_tn iry = ip->ry[i];
		real_tn irz = ip->rz[i];
		real_tn ivx = ip->vx[i];
		real_tn ivy = ip->vy[i];
		real_tn ivz = ip->vz[i];
		real_tn ipnax = ip->pnax[i];
		real_tn ipnay = ip->pnay[i];
		real_tn ipnaz = ip->pnaz[i];
		#pragma unroll 1
		for (uint_t k = 0; k < SIMD; ++k) {
			#pragma unroll 1
			for (uint_t j = 0; j < LSIZE; ++j) {
				real_tn ee = iee + jp->e2[j];
				real_tn rx = irx - jp->rx[j];
				real_tn ry = iry - jp->ry[j];
				real_tn rz = irz - jp->rz[j];
				real_tn vx = ivx - jp->vx[j];
				real_tn vy = ivy - jp->vy[j];
				real_tn vz = ivz - jp->vz[j];

				real_tn rr = ee;
				rr += rx * rx + ry * ry + rz * rz;

				real_tn inv_r1 = rsqrt(rr);
				inv_r1 = (rr > ee) ? (inv_r1):(0);
				real_tn inv_r = inv_r1;
				real_tn inv_r2 = inv_r * inv_r;

//				real_tn im = ip->m[i];
				real_tn im2 = im * im;
				real_tn im_r = im * inv_r;
				real_tn iv2 = ivx * ivx
							+ ivy * ivy
							+ ivz * ivz;
				real_tn iv4 = iv2 * iv2;
				real_tn niv = rx * ivx
							+ ry * ivy
							+ rz * ivz;
				niv *= inv_r;
				real_tn niv2 = niv * niv;

				real_tn jm = jp->m[j];
				real_tn jm2 = jm * jm;
				real_tn jm_r = jm * inv_r;
				real_tn jv2 = jp->vx[j] * jp->vx[j]
							+ jp->vy[j] * jp->vy[j]
							+ jp->vz[j] * jp->vz[j];
				real_tn jv4 = jv2 * jv2;
				real_tn njv = rx * jp->vx[j]
							+ ry * jp->vy[j]
							+ rz * jp->vz[j];
				njv *= inv_r;
				real_tn njv2 = njv * njv;

				real_tn imjm = im * jm;
				real_tn vv = vx * vx
						   + vy * vy
						   + vz * vz;
				real_tn ivjv = ivx * jp->vx[j]
							 + ivy * jp->vy[j]
							 + ivz * jp->vz[j];
				real_tn nv = rx * vx
						   + ry * vy
						   + rz * vz;
				nv *= inv_r;
				real_tn nvnv = nv * nv;
				real_tn nivnjv = niv * njv;

				uint_t order = clight.order;
				real_t inv_c = clight.inv1;

				real_tn jpnA = pnterms_A(im, im2, im_r, iv2, iv4, +niv, niv2,
										 jm, jm2, jm_r, jv2, jv4, +njv, njv2,
										 imjm, inv_r, inv_r2, vv, ivjv,
										 nv, nvnv, nivnjv, order, inv_c);

				real_tn jpnB = pnterms_B(im, im2, im_r, iv2, iv4, +niv, niv2,
										 jm, jm2, jm_r, jv2, jv4, +njv, njv2,
										 imjm, inv_r, inv_r2, vv, ivjv,
										 nv, nvnv, nivnjv, order, inv_c);

				ipnax += jpnA * rx + jpnB * vx;
				ipnay += jpnA * ry + jpnB * vy;
				ipnaz += jpnA * rz + jpnB * vz;
			}
			shuff(im, SIMD);
			shuff(iee, SIMD);
			shuff(irx, SIMD);
			shuff(iry, SIMD);
			shuff(irz, SIMD);
			shuff(ivx, SIMD);
			shuff(ivy, SIMD);
			shuff(ivz, SIMD);
			shuff(ipnax, SIMD);
			shuff(ipnay, SIMD);
			shuff(ipnaz, SIMD);
		}
		ip->pnax[i] = ipnax;
		ip->pnay[i] = ipnay;
		ip->pnaz[i] = ipnaz;
	}
}


static inline void
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
	for (uint_t ii = LSIZE * SIMD * get_group_id(0);
				ii < ni;
				ii += LSIZE * SIMD * get_num_groups(0)) {
		uint_t iN = min((uint_t)(LSIZE * SIMD), (ni - ii));
		ip->m[get_local_id(0)] = (real_tn)(0);
		ip->e2[get_local_id(0)] = (real_tn)(0);
		barrier(CLK_LOCAL_MEM_FENCE);
		async_work_group_copy(ip->_m, __im+ii, iN, 0);
		async_work_group_copy(ip->_e2, __ie2+ii, iN, 0);
		async_work_group_copy(ip->_rx, __irdot+(0*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_ry, __irdot+(0*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_rz, __irdot+(0*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_vx, __irdot+(1*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_vy, __irdot+(1*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_vz, __irdot+(1*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_pnax, __ipnacc+(0*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_pnay, __ipnacc+(0*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_pnaz, __ipnacc+(0*NDIM+2)*ni+ii, iN, 0);
		for (uint_t jj = 0;
					jj < nj;
					jj += LSIZE * SIMD) {
			uint_t jN = min((uint_t)(LSIZE * SIMD), (nj - jj));
			jp->m[get_local_id(0)] = (real_tn)(0);
			jp->e2[get_local_id(0)] = (real_tn)(0);
			barrier(CLK_LOCAL_MEM_FENCE);
			async_work_group_copy(jp->_m, __jm+jj, jN, 0);
			async_work_group_copy(jp->_e2, __je2+jj, jN, 0);
			async_work_group_copy(jp->_rx, __jrdot+(0*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_ry, __jrdot+(0*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_rz, __jrdot+(0*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_vx, __jrdot+(1*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_vy, __jrdot+(1*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_vz, __jrdot+(1*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_pnax, __jpnacc+(0*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_pnay, __jpnacc+(0*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_pnaz, __jpnacc+(0*NDIM+2)*nj+jj, jN, 0);
			barrier(CLK_LOCAL_MEM_FENCE);
			pnacc_kernel_core(ip, jp, clight);
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		async_work_group_copy(__ipnacc+(0*NDIM+0)*ni+ii, ip->_pnax, iN, 0);
		async_work_group_copy(__ipnacc+(0*NDIM+1)*ni+ii, ip->_pnay, iN, 0);
		async_work_group_copy(__ipnacc+(0*NDIM+2)*ni+ii, ip->_pnaz, iN, 0);
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

