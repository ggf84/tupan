#include "acc_kernel_common.h"


static inline void
p2p_acc_kernel_core(
	uint_t lid,
	local Acc_Data *ip,
	local Acc_Data *jp)
// flop count: 28
{
	uint_t jwarp = lid / WARP;
	uint_t jlane = lid % WARP;
	uint_t jlid = WARP * jwarp + jlane;

	for (uint_t k = 0; k < SIMD; ++k) {
		for (uint_t w = 0; w < WGSIZE/WARP; ++w) {
			uint_t iwarp = jwarp^w;
			for (uint_t l = 0; l < WARP; ++l) {
				uint_t ilane = jlane^l;
				uint_t ilid = WARP * iwarp + ilane;
				for (uint_t ii = 0; ii < LMSIZE; ii += WGSIZE) {
					uint_t i = ii + ilid;
					real_tn iax = ip->ax[i];
					real_tn iay = ip->ay[i];
					real_tn iaz = ip->az[i];
					for (uint_t jj = 0; jj < LMSIZE; jj += WGSIZE) {
						uint_t j = jj + jlid;
						real_tn ee = ip->e2[i] + jp->e2[j];
						real_tn rx = ip->rx[i] - jp->rx[j];
						real_tn ry = ip->ry[i] - jp->ry[j];
						real_tn rz = ip->rz[i] - jp->rz[j];

						real_tn rr = ee;
						rr += rx * rx + ry * ry + rz * rz;

						real_tn inv_r3 = rsqrt(rr);
						inv_r3 *= inv_r3 * inv_r3;

						real_tn im_r3 = ip->m[i] * inv_r3;
						jp->ax[j] += im_r3 * rx;
						jp->ay[j] += im_r3 * ry;
						jp->az[j] += im_r3 * rz;

						real_tn jm_r3 = jp->m[j] * inv_r3;
						iax -= jm_r3 * rx;
						iay -= jm_r3 * ry;
						iaz -= jm_r3 * rz;
					}
					ip->ax[i] = iax;
					ip->ay[i] = iay;
					ip->az[i] = iaz;
				}
			}
		}
		for (uint_t jj = 0; jj < LMSIZE; jj += WGSIZE) {
			uint_t j = jj + jlid;
			shuff(jp->m[j], SIMD);
			shuff(jp->e2[j], SIMD);
			shuff(jp->rx[j], SIMD);
			shuff(jp->ry[j], SIMD);
			shuff(jp->rz[j], SIMD);
			shuff(jp->ax[j], SIMD);
			shuff(jp->ay[j], SIMD);
			shuff(jp->az[j], SIMD);
		}
	}
}


static inline void
acc_kernel_core(
	uint_t lid,
	local Acc_Data *ip,
	local Acc_Data *jp)
// flop count: 21
{
	uint_t jwarp = lid / WARP;
	uint_t jlane = lid % WARP;
	uint_t jlid = WARP * jwarp + jlane;

	for (uint_t k = 0; k < SIMD; ++k) {
		for (uint_t w = 0; w < WGSIZE/WARP; ++w) {
			uint_t iwarp = jwarp^w;
			for (uint_t l = 0; l < WARP; ++l) {
				uint_t ilane = jlane^l;
				uint_t ilid = WARP * iwarp + ilane;
				for (uint_t ii = 0; ii < LMSIZE; ii += WGSIZE) {
					uint_t i = ii + ilid;
//					real_tn iax = ip->ax[i];
//					real_tn iay = ip->ay[i];
//					real_tn iaz = ip->az[i];
					for (uint_t jj = 0; jj < LMSIZE; jj += WGSIZE) {
						uint_t j = jj + jlid;
						real_tn ee = ip->e2[i] + jp->e2[j];
						real_tn rx = ip->rx[i] - jp->rx[j];
						real_tn ry = ip->ry[i] - jp->ry[j];
						real_tn rz = ip->rz[i] - jp->rz[j];

						real_tn rr = ee;
						rr += rx * rx + ry * ry + rz * rz;

						real_tn inv_r3 = rsqrt(rr);
						inv_r3 = (rr > ee) ? (inv_r3):(0);
						inv_r3 *= inv_r3 * inv_r3;

						real_tn im_r3 = ip->m[i] * inv_r3;
						jp->ax[j] += im_r3 * rx;
						jp->ay[j] += im_r3 * ry;
						jp->az[j] += im_r3 * rz;

//						real_tn jm_r3 = jp->m[j] * inv_r3;
//						iax -= jm_r3 * rx;
//						iay -= jm_r3 * ry;
//						iaz -= jm_r3 * rz;
					}
//					ip->ax[i] = iax;
//					ip->ay[i] = iay;
//					ip->az[i] = iaz;
				}
			}
		}
		for (uint_t kk = 0; kk < LMSIZE; kk += WGSIZE) {
			uint_t j = kk + lid;
			shuff(jp->m[j], SIMD);
			shuff(jp->e2[j], SIMD);
			shuff(jp->rx[j], SIMD);
			shuff(jp->ry[j], SIMD);
			shuff(jp->rz[j], SIMD);
			shuff(jp->ax[j], SIMD);
			shuff(jp->ay[j], SIMD);
			shuff(jp->az[j], SIMD);
		}
	}
}


// ----------------------------------------------------------------------------


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
	uint_t block = LMSIZE * SIMD;
	uint_t lid = get_local_id(0);
	uint_t igrp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t grpsize = ngrps * block;
	for (uint_t iii = 0, ii = igrp * block;
				iii < ni;
				iii += grpsize, ii += grpsize) {
		uint_t iN = min(block, (ni - ii));
		uint_t imask = ii < ni;
		if (imask) {
			zero_Acc_Data(lid, ip);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (imask) {
			async_work_group_copy(ip->_m, __im+ii, iN, 0);
			async_work_group_copy(ip->_e2, __ie2+ii, iN, 0);
			async_work_group_copy(ip->_rx, __irdot+(0*NDIM+0)*ni+ii, iN, 0);
			async_work_group_copy(ip->_ry, __irdot+(0*NDIM+1)*ni+ii, iN, 0);
			async_work_group_copy(ip->_rz, __irdot+(0*NDIM+2)*ni+ii, iN, 0);
		}

		for (uint_t jjj = 0, jj = 0;
					jjj < nj;
					jjj += block, jj += block) {
			uint_t jN = min(block, (nj - jj));
			uint_t jmask = jj < nj;
			if (imask && jmask) {
				zero_Acc_Data(lid, jp);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			if (imask && jmask) {
				async_work_group_copy(jp->_m, __jm+jj, jN, 0);
				async_work_group_copy(jp->_e2, __je2+jj, jN, 0);
				async_work_group_copy(jp->_rx, __jrdot+(0*NDIM+0)*nj+jj, jN, 0);
				async_work_group_copy(jp->_ry, __jrdot+(0*NDIM+1)*nj+jj, jN, 0);
				async_work_group_copy(jp->_rz, __jrdot+(0*NDIM+2)*nj+jj, jN, 0);
			}

			barrier(CLK_LOCAL_MEM_FENCE);
			if (imask && jmask) {
				acc_kernel_core(lid, jp, ip);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		if (imask) {
			for (uint_t kk = 0, k = lid;
						kk < iN;
						kk += WGSIZE, k += WGSIZE) {
				if (k < iN) {
					(__iadot+(0*NDIM+0)*ni+ii)[k] += ip->_ax[k];
					(__iadot+(0*NDIM+1)*ni+ii)[k] += ip->_ay[k];
					(__iadot+(0*NDIM+2)*ni+ii)[k] += ip->_az[k];
				}
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
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
	uint_t block = LMSIZE * SIMD;
	uint_t lid = get_local_id(0);
	uint_t igrp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t grpsize = ngrps * block;
	for (uint_t iii = 0, ii = igrp * block;
				iii < ni;
				iii += grpsize, ii += grpsize) {
		uint_t iN = min(block, (ni - ii));
		uint_t imask = ii < ni;
		local Acc_Data ip;
		if (imask) {
			zero_Acc_Data(lid, &ip);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (imask) {
			async_work_group_copy(ip._m, __im+ii, iN, 0);
			async_work_group_copy(ip._e2, __ie2+ii, iN, 0);
			async_work_group_copy(ip._rx, __irdot+(0*NDIM+0)*ni+ii, iN, 0);
			async_work_group_copy(ip._ry, __irdot+(0*NDIM+1)*ni+ii, iN, 0);
			async_work_group_copy(ip._rz, __irdot+(0*NDIM+2)*ni+ii, iN, 0);
		}

		for (uint_t g = 0; g < ngrps; ++g) {
			uint_t jgrp = (igrp + g) % ngrps;
			for (uint_t jjj = 0, jj = jgrp * block;
						jjj < nj;
						jjj += grpsize, jj += grpsize) {
				uint_t jN = min(block, (nj - jj));
				uint_t jmask = jj < nj;
				local Acc_Data jp;
				if (imask && jmask) {
					zero_Acc_Data(lid, &jp);
				}
				barrier(CLK_LOCAL_MEM_FENCE);
				if (imask && jmask) {
					async_work_group_copy(jp._m, __jm+jj, jN, 0);
					async_work_group_copy(jp._e2, __je2+jj, jN, 0);
					async_work_group_copy(jp._rx, __jrdot+(0*NDIM+0)*nj+jj, jN, 0);
					async_work_group_copy(jp._ry, __jrdot+(0*NDIM+1)*nj+jj, jN, 0);
					async_work_group_copy(jp._rz, __jrdot+(0*NDIM+2)*nj+jj, jN, 0);
				}

				barrier(CLK_LOCAL_MEM_FENCE);
				if (imask && jmask) {
					p2p_acc_kernel_core(lid, &jp, &ip);
				}
				barrier(CLK_LOCAL_MEM_FENCE);

				if (imask && jmask) {
					for (uint_t kk = 0, k = lid;
								kk < jN;
								kk += WGSIZE, k += WGSIZE) {
						if (k < jN) {
							atomic_fadd(&(__jadot+(0*NDIM+0)*nj+jj)[k], jp._ax[k]);
							atomic_fadd(&(__jadot+(0*NDIM+1)*nj+jj)[k], jp._ay[k]);
							atomic_fadd(&(__jadot+(0*NDIM+2)*nj+jj)[k], jp._az[k]);
						}
					}
				}
				barrier(CLK_GLOBAL_MEM_FENCE);
			}
		}

		if (imask) {
			for (uint_t kk = 0, k = lid;
						kk < iN;
						kk += WGSIZE, k += WGSIZE) {
				if (k < iN) {
					(__iadot+(0*NDIM+0)*ni+ii)[k] += ip._ax[k];
					(__iadot+(0*NDIM+1)*ni+ii)[k] += ip._ay[k];
					(__iadot+(0*NDIM+2)*ni+ii)[k] += ip._az[k];
				}
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}


// ----------------------------------------------------------------------------


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
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

/*
kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
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
*/
