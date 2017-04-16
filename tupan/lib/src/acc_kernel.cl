#include "acc_kernel_common.h"


static inline void
p2p_acc_kernel_core(
	uint_t lane,
	local Acc_Data *ip,
	local Acc_Data *jp)
// flop count: 28
{
	for (uint_t l = 0; l < NLANES; ++l) {
		for (uint_t k = 0; k < SIMD; ++k) {
			for (uint_t ii = 0, i = lane;
						ii < LMSIZE;
						ii += NLANES, i += NLANES) {
				for (uint_t jj = 0, j = lane^l;
							jj < LMSIZE;
							jj += NLANES, j += NLANES) {
					real_tn ee = ip->e2[i] + jp->e2[j];
					real_tn rx = ip->rx[i] - jp->rx[j];
					real_tn ry = ip->ry[i] - jp->ry[j];
					real_tn rz = ip->rz[i] - jp->rz[j];

					real_tn rr = ee;
					rr += rx * rx;
					rr += ry * ry;
					rr += rz * rz;

					real_tn inv_r3 = rsqrt(rr);
					inv_r3 *= inv_r3 * inv_r3;

					real_tn im_r3 = ip->m[i] * inv_r3;
					jp->ax[j] += im_r3 * rx;
					jp->ay[j] += im_r3 * ry;
					jp->az[j] += im_r3 * rz;

					real_tn jm_r3 = jp->m[j] * inv_r3;
					ip->ax[i] += jm_r3 * rx;
					ip->ay[i] += jm_r3 * ry;
					ip->az[i] += jm_r3 * rz;
				}
			}
			simd_shuff_Acc_Data(lane, ip);
		}
	}
}


static inline void
acc_kernel_core(
	uint_t lane,
	local Acc_Data *ip,
	local Acc_Data *jp)
// flop count: 21
{
	for (uint_t l = 0; l < NLANES; ++l) {
		for (uint_t k = 0; k < SIMD; ++k) {
			for (uint_t ii = 0, i = lane;
						ii < LMSIZE;
						ii += NLANES, i += NLANES) {
				for (uint_t jj = 0, j = l;
							jj < LMSIZE;
							jj += NLANES, j += NLANES) {
					real_tn ee = ip->e2[i] + jp->e2[j];
					real_tn rx = ip->rx[i] - jp->rx[j];
					real_tn ry = ip->ry[i] - jp->ry[j];
					real_tn rz = ip->rz[i] - jp->rz[j];

					real_tn rr = ee;
					rr += rx * rx;
					rr += ry * ry;
					rr += rz * rz;

					real_tn inv_r3 = rsqrt(rr);
					inv_r3 = (rr > ee) ? (inv_r3):(0);
					inv_r3 *= inv_r3 * inv_r3;

//					real_tn im_r3 = ip->m[i] * inv_r3;
//					jp->ax[j] += im_r3 * rx;
//					jp->ay[j] += im_r3 * ry;
//					jp->az[j] += im_r3 * rz;

					real_tn jm_r3 = jp->m[j] * inv_r3;
					ip->ax[i] += jm_r3 * rx;
					ip->ay[i] += jm_r3 * ry;
					ip->az[i] += jm_r3 * rz;
				}
			}
			simd_shuff_Acc_Data(lane, ip);
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
	local Acc_Data ip[NWARPS];
	local Acc_Data jp[NWARPS];
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t lane = lid % NLANES;
	uint_t warp = lid / NLANES;
	for (uint_t ii = grp;
				ii * NWARPS * LMSIZE * SIMD < ni;
				ii += ngrps) {
		uint_t ibid = (ii * NWARPS + warp) * LMSIZE * SIMD;
		setup_Acc_Data(
			ibid, lane, &ip[warp],
			ni, __im, __ie2, __irdot
		);

		for (uint_t jj = 0;
					jj * NWARPS * LMSIZE * SIMD < nj;
					jj += 1) {
			uint_t jbid = (jj * NWARPS + warp) * LMSIZE * SIMD;
			setup_Acc_Data(
				jbid, lane, &jp[warp],
				nj, __jm, __je2, __jrdot
			);

			for (uint_t w = 0; w < NWARPS; ++w) {
				p2p_acc_kernel_core(lane, &ip[warp^w], &jp[warp]);
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			for (uint_t kk = 0, k = lane;
						kk < LMSIZE * SIMD;
						kk += NLANES, k += NLANES) {
				if (jbid+k < nj) {
					atomic_fadd(&(__jadot+(0*NDIM+0)*nj)[jbid+k], jp[warp]._ax[k]);
					atomic_fadd(&(__jadot+(0*NDIM+1)*nj)[jbid+k], jp[warp]._ay[k]);
					atomic_fadd(&(__jadot+(0*NDIM+2)*nj)[jbid+k], jp[warp]._az[k]);
				}
			}
		}

		for (uint_t kk = 0, k = lane;
					kk < LMSIZE * SIMD;
					kk += NLANES, k += NLANES) {
			if (ibid+k < ni) {
				(__iadot+(0*NDIM+0)*ni)[ibid+k] -= ip[warp]._ax[k];
				(__iadot+(0*NDIM+1)*ni)[ibid+k] -= ip[warp]._ay[k];
				(__iadot+(0*NDIM+2)*ni)[ibid+k] -= ip[warp]._az[k];
			}
		}
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
acc_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	global real_t __iadot[])
{
	// ------------------------------------------------------------------------
	const uint_t nj = ni;
	global const real_t *__jm = __im;
	global const real_t *__je2 = __ie2;
	global const real_t *__jrdot = __irdot;
	global real_t *__jadot = __iadot;
	// ------------------------------------------------------------------------

	local Acc_Data ip[NWARPS];
	local Acc_Data jp[NWARPS];
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t lane = lid % NLANES;
	uint_t warp = lid / NLANES;
	for (uint_t ii = grp;
				ii * NWARPS * LMSIZE * SIMD < ni;
				ii += ngrps) {
		uint_t ibid = (ii * NWARPS + warp) * LMSIZE * SIMD;
		setup_Acc_Data(
			ibid, lane, &ip[warp],
			ni, __im, __ie2, __irdot
		);

		for (uint_t jj = 0;
					jj * NWARPS * LMSIZE * SIMD < nj;
					jj += 1) {
			uint_t jbid = (jj * NWARPS + warp) * LMSIZE * SIMD;
			setup_Acc_Data(
				jbid, lane, &jp[warp],
				nj, __jm, __je2, __jrdot
			);

			for (uint_t w = 0; w < NWARPS; ++w) {
				acc_kernel_core(lane, &ip[warp^w], &jp[warp]);
				barrier(CLK_LOCAL_MEM_FENCE);
			}

//			for (uint_t kk = 0, k = lane;
//						kk < LMSIZE * SIMD;
//						kk += NLANES, k += NLANES) {
//				if (jbid+k < nj) {
//					atomic_fadd(&(__jadot+(0*NDIM+0)*nj)[jbid+k], jp[warp]._ax[k]);
//					atomic_fadd(&(__jadot+(0*NDIM+1)*nj)[jbid+k], jp[warp]._ay[k]);
//					atomic_fadd(&(__jadot+(0*NDIM+2)*nj)[jbid+k], jp[warp]._az[k]);
//				}
//			}
		}

		for (uint_t kk = 0, k = lane;
					kk < LMSIZE * SIMD;
					kk += NLANES, k += NLANES) {
			if (ibid+k < ni) {
				(__iadot+(0*NDIM+0)*ni)[ibid+k] -= ip[warp]._ax[k];
				(__iadot+(0*NDIM+1)*ni)[ibid+k] -= ip[warp]._ay[k];
				(__iadot+(0*NDIM+2)*ni)[ibid+k] -= ip[warp]._az[k];
			}
		}
	}
}

