#include "acc_kernel_common.h"


static inline void
p2p_acc_kernel_core(
	uint_t warp_id,
	uint_t lane_id,
	local Acc_Data *ip,
	local Acc_Data *jp)
// flop count: 28
{
	for (uint_t k = 0; k < SIMD; ++k)
	for (uint_t l = 0; l < WGSIZE; ++l)
	for (uint_t ii = 0, i = (NLANES * warp_id + lane_id);
				ii < LMSIZE;
				ii += WGSIZE, i += WGSIZE) {
		real_tn iax = (real_tn)(0);
		real_tn iay = (real_tn)(0);
		real_tn iaz = (real_tn)(0);
		for (uint_t jj = 0, j = (NLANES * warp_id + lane_id)^l;
					jj < LMSIZE;
					jj += WGSIZE, j += WGSIZE) {
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
		ip->ax[i] += iax;
		ip->ay[i] += iay;
		ip->az[i] += iaz;

		shuff(ip->m[i], SIMD);
		shuff(ip->e2[i], SIMD);
		shuff(ip->rx[i], SIMD);
		shuff(ip->ry[i], SIMD);
		shuff(ip->rz[i], SIMD);
		shuff(ip->ax[i], SIMD);
		shuff(ip->ay[i], SIMD);
		shuff(ip->az[i], SIMD);
	}
}


static inline void
acc_kernel_core(
	uint_t warp_id,
	uint_t lane_id,
	local Acc_Data *ip,
	local Acc_Data *jp)
// flop count: 21
{
	for (uint_t k = 0; k < SIMD; ++k)
	for (uint_t l = 0; l < NLANES; ++l)
	for (uint_t ii = 0, i = (lane_id)^l;
				ii < LMSIZE;
				ii += NLANES, i += NLANES) {
//		real_tn iax = (real_tn)(0);
//		real_tn iay = (real_tn)(0);
//		real_tn iaz = (real_tn)(0);
		for (uint_t jj = 0, j = (NLANES * warp_id + lane_id);
					jj < LMSIZE;
					jj += WGSIZE, j += WGSIZE) {
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

//			real_tn jm_r3 = jp->m[j] * inv_r3;
//			iax -= jm_r3 * rx;
//			iay -= jm_r3 * ry;
//			iaz -= jm_r3 * rz;
		}
//		ip->ax[i] += iax;
//		ip->ay[i] += iay;
//		ip->az[i] += iaz;

		shuff(ip->m[i], SIMD);
		shuff(ip->e2[i], SIMD);
		shuff(ip->rx[i], SIMD);
		shuff(ip->ry[i], SIMD);
		shuff(ip->rz[i], SIMD);
		shuff(ip->ax[i], SIMD);
		shuff(ip->ay[i], SIMD);
		shuff(ip->az[i], SIMD);
	}
}


// ----------------------------------------------------------------------------


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
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
	global real_t __jadot[])
{
	local Acc_Data ip;
	local Acc_Data jp;
	uint_t block = LMSIZE * SIMD;
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t warp_id = lid / NLANES;
	uint_t lane_id = lid % NLANES;
	for (uint_t ii = grp * block;
				ii < ni;
				ii += ngrps * block) {
		barrier(CLK_LOCAL_MEM_FENCE);
		zero_Acc_Data(warp_id, lane_id, &ip);
		barrier(CLK_LOCAL_MEM_FENCE);
		uint_t iN = min(block, (ni - ii));
		async_work_group_copy(ip._m, __im+ii, iN, 0);
		async_work_group_copy(ip._e2, __ie2+ii, iN, 0);
		async_work_group_copy(ip._rx, __irdot+(0*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip._ry, __irdot+(0*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip._rz, __irdot+(0*NDIM+2)*ni+ii, iN, 0);

		for (uint_t jj = 0;
					jj < nj;
					jj += block) {
			barrier(CLK_LOCAL_MEM_FENCE);
			zero_Acc_Data(warp_id, lane_id, &jp);
			barrier(CLK_LOCAL_MEM_FENCE);
			uint_t jN = min(block, (nj - jj));
			async_work_group_copy(jp._m, __jm+jj, jN, 0);
			async_work_group_copy(jp._e2, __je2+jj, jN, 0);
			async_work_group_copy(jp._rx, __jrdot+(0*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp._ry, __jrdot+(0*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp._rz, __jrdot+(0*NDIM+2)*nj+jj, jN, 0);

			barrier(CLK_LOCAL_MEM_FENCE);
			acc_kernel_core(warp_id, lane_id, &jp, &ip);
			barrier(CLK_LOCAL_MEM_FENCE);

//			for (uint_t kk = 0, k = NLANES * warp_id + lane_id;
//						kk < block;
//						kk += WGSIZE, k += WGSIZE) {
//				if (k < jN) {
//					atomic_fadd(&(__jadot+(0*NDIM+0)*nj)[jj+k], jp._ax[k]);
//					atomic_fadd(&(__jadot+(0*NDIM+1)*nj)[jj+k], jp._ay[k]);
//					atomic_fadd(&(__jadot+(0*NDIM+2)*nj)[jj+k], jp._az[k]);
//				}
//			}
		}

		for (uint_t kk = 0, k = NLANES * warp_id + lane_id;
					kk < block;
					kk += WGSIZE, k += WGSIZE) {
			if (k < iN) {
				(__iadot+(0*NDIM+0)*ni)[ii+k] += ip._ax[k];
				(__iadot+(0*NDIM+1)*ni)[ii+k] += ip._ay[k];
				(__iadot+(0*NDIM+2)*ni)[ii+k] += ip._az[k];
			}
		}
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
	local Acc_Data ip;
	local Acc_Data jp;
	uint_t block = LMSIZE * SIMD;
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t warp_id = lid / NLANES;
	uint_t lane_id = lid % NLANES;
	for (uint_t ii = grp * block;
				ii < ni;
				ii += ngrps * block) {
		barrier(CLK_LOCAL_MEM_FENCE);
		zero_Acc_Data(warp_id, lane_id, &ip);
		barrier(CLK_LOCAL_MEM_FENCE);
		uint_t iN = min(block, (ni - ii));
		async_work_group_copy(ip._m, __im+ii, iN, 0);
		async_work_group_copy(ip._e2, __ie2+ii, iN, 0);
		async_work_group_copy(ip._rx, __irdot+(0*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip._ry, __irdot+(0*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip._rz, __irdot+(0*NDIM+2)*ni+ii, iN, 0);

		for (uint_t jj = 0;
					jj < nj;
					jj += block) {
			barrier(CLK_LOCAL_MEM_FENCE);
			zero_Acc_Data(warp_id, lane_id, &jp);
			barrier(CLK_LOCAL_MEM_FENCE);
			uint_t jN = min(block, (nj - jj));
			async_work_group_copy(jp._m, __jm+jj, jN, 0);
			async_work_group_copy(jp._e2, __je2+jj, jN, 0);
			async_work_group_copy(jp._rx, __jrdot+(0*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp._ry, __jrdot+(0*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp._rz, __jrdot+(0*NDIM+2)*nj+jj, jN, 0);

			barrier(CLK_LOCAL_MEM_FENCE);
			p2p_acc_kernel_core(warp_id, lane_id, &jp, &ip);
			barrier(CLK_LOCAL_MEM_FENCE);

			for (uint_t kk = 0, k = NLANES * warp_id + lane_id;
						kk < block;
						kk += WGSIZE, k += WGSIZE) {
				if (k < jN) {
					atomic_fadd(&(__jadot+(0*NDIM+0)*nj)[jj+k], jp._ax[k]);
					atomic_fadd(&(__jadot+(0*NDIM+1)*nj)[jj+k], jp._ay[k]);
					atomic_fadd(&(__jadot+(0*NDIM+2)*nj)[jj+k], jp._az[k]);
				}
			}
		}

		for (uint_t kk = 0, k = NLANES * warp_id + lane_id;
					kk < block;
					kk += WGSIZE, k += WGSIZE) {
			if (k < iN) {
				(__iadot+(0*NDIM+0)*ni)[ii+k] += ip._ax[k];
				(__iadot+(0*NDIM+1)*ni)[ii+k] += ip._ay[k];
				(__iadot+(0*NDIM+2)*ni)[ii+k] += ip._az[k];
			}
		}
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
	acc_kernel_impl(
		ni, __im, __ie2, __irdot,
		ni, __im, __ie2, __irdot,
		__iadot, __iadot
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
	acc_kernel_impl(
		ni, __im, __ie2, __irdot,
		nj, __jm, __je2, __jrdot,
		__iadot, __jadot
	);
	acc_kernel_impl(
		nj, __jm, __je2, __jrdot,
		ni, __im, __ie2, __irdot,
		__jadot, __iadot
	);
}
*/
