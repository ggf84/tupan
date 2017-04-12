#include "acc_kernel_common.h"


static inline void
p2p_acc_kernel_core(
	uint_t warp_id,
	uint_t lane_id,
	local Acc_Data *ip,
	local Acc_Data *jp)
// flop count: 28
{
	for (uint_t w = 0; w < NWARPS; ++w) {
		for (uint_t l = 0; l < NLANES; ++l) {
			for (uint_t k = 0; k < SIMD; ++k) {
				for (uint_t ii = 0, i = NLANES * warp_id + lane_id;
							ii < LMSIZE;
							ii += WGSIZE, i += WGSIZE) {
					for (uint_t jj = 0, j = NLANES * (warp_id^w) + (lane_id^l);
								jj < LMSIZE;
								jj += WGSIZE, j += WGSIZE) {
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
				simd_shuff_Acc_Data(warp_id, lane_id, ip);
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
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
	for (uint_t w = 0; w < NWARPS; ++w) {
		for (uint_t l = 0; l < NLANES; ++l) {
			for (uint_t k = 0; k < SIMD; ++k) {
				for (uint_t ii = 0, i = NLANES * warp_id + lane_id;
							ii < LMSIZE;
							ii += WGSIZE, i += WGSIZE) {
					for (uint_t jj = 0, j = NLANES * (w) + (l);
								jj < LMSIZE;
								jj += WGSIZE, j += WGSIZE) {
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

//						real_tn im_r3 = ip->m[i] * inv_r3;
//						jp->ax[j] += im_r3 * rx;
//						jp->ay[j] += im_r3 * ry;
//						jp->az[j] += im_r3 * rz;

						real_tn jm_r3 = jp->m[j] * inv_r3;
						ip->ax[i] += jm_r3 * rx;
						ip->ay[i] += jm_r3 * ry;
						ip->az[i] += jm_r3 * rz;
					}
				}
				simd_shuff_Acc_Data(warp_id, lane_id, ip);
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
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
		uint_t iN = min(block, (ni - ii));
		zero_Acc_Data(warp_id, lane_id, &ip);
		async_work_group_copy(ip._m, __im+ii, iN, 0);
		async_work_group_copy(ip._e2, __ie2+ii, iN, 0);
		async_work_group_copy(ip._rx, __irdot+(0*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip._ry, __irdot+(0*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip._rz, __irdot+(0*NDIM+2)*ni+ii, iN, 0);
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint_t jj = 0;
					jj < nj;
					jj += block) {
			uint_t jN = min(block, (nj - jj));
			zero_Acc_Data(warp_id, lane_id, &jp);
			async_work_group_copy(jp._m, __jm+jj, jN, 0);
			async_work_group_copy(jp._e2, __je2+jj, jN, 0);
			async_work_group_copy(jp._rx, __jrdot+(0*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp._ry, __jrdot+(0*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp._rz, __jrdot+(0*NDIM+2)*nj+jj, jN, 0);
			barrier(CLK_LOCAL_MEM_FENCE);

			acc_kernel_core(warp_id, lane_id, &ip, &jp);

//			for (uint_t kk = 0, k = NLANES * warp_id + lane_id;
//						kk < block;
//						kk += WGSIZE, k += WGSIZE) {
//				if (k < jN) {
//					atomic_fadd(&(__jadot+(0*NDIM+0)*nj)[jj+k], jp._ax[k]);
//					atomic_fadd(&(__jadot+(0*NDIM+1)*nj)[jj+k], jp._ay[k]);
//					atomic_fadd(&(__jadot+(0*NDIM+2)*nj)[jj+k], jp._az[k]);
//				}
//			}
//			barrier(CLK_GLOBAL_MEM_FENCE);
		}

		for (uint_t kk = 0, k = NLANES * warp_id + lane_id;
					kk < block;
					kk += WGSIZE, k += WGSIZE) {
			if (k < iN) {
				(__iadot+(0*NDIM+0)*ni)[ii+k] -= ip._ax[k];
				(__iadot+(0*NDIM+1)*ni)[ii+k] -= ip._ay[k];
				(__iadot+(0*NDIM+2)*ni)[ii+k] -= ip._az[k];
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
acc_kernel_rectangle__(
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
		uint_t iN = min(block, (ni - ii));
		zero_Acc_Data(warp_id, lane_id, &ip);
		async_work_group_copy(ip._m, __im+ii, iN, 0);
		async_work_group_copy(ip._e2, __ie2+ii, iN, 0);
		async_work_group_copy(ip._rx, __irdot+(0*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip._ry, __irdot+(0*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip._rz, __irdot+(0*NDIM+2)*ni+ii, iN, 0);
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint_t jj = 0;
					jj < nj;
					jj += block) {
			uint_t jN = min(block, (nj - jj));
			zero_Acc_Data(warp_id, lane_id, &jp);
			async_work_group_copy(jp._m, __jm+jj, jN, 0);
			async_work_group_copy(jp._e2, __je2+jj, jN, 0);
			async_work_group_copy(jp._rx, __jrdot+(0*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp._ry, __jrdot+(0*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp._rz, __jrdot+(0*NDIM+2)*nj+jj, jN, 0);
			barrier(CLK_LOCAL_MEM_FENCE);

			p2p_acc_kernel_core(warp_id, lane_id, &ip, &jp);

			for (uint_t kk = 0, k = NLANES * warp_id + lane_id;
						kk < block;
						kk += WGSIZE, k += WGSIZE) {
				if (k < jN) {
					atomic_fadd(&(__jadot+(0*NDIM+0)*nj)[jj+k], jp._ax[k]);
					atomic_fadd(&(__jadot+(0*NDIM+1)*nj)[jj+k], jp._ay[k]);
					atomic_fadd(&(__jadot+(0*NDIM+2)*nj)[jj+k], jp._az[k]);
				}
			}
			barrier(CLK_GLOBAL_MEM_FENCE);
		}

		for (uint_t kk = 0, k = NLANES * warp_id + lane_id;
					kk < block;
					kk += WGSIZE, k += WGSIZE) {
			if (k < iN) {
				(__iadot+(0*NDIM+0)*ni)[ii+k] -= ip._ax[k];
				(__iadot+(0*NDIM+1)*ni)[ii+k] -= ip._ay[k];
				(__iadot+(0*NDIM+2)*ni)[ii+k] -= ip._az[k];
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}


// ----------------------------------------------------------------------------


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
acc_kernel_triangle__(
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






// ----------------------------------------------------------------------------





static inline void
setup_Acc_Data(
	uint_t bid,
	uint_t lane_id,
	local Acc_Data *p,
	uint_t n,
	global const real_t __m[],
	global const real_t __e2[],
	global const real_t __rdot[])
{
	for (uint_t kk = 0, k = lane_id;
				kk < LMSIZE * SIMD;
				kk += NLANES, k += NLANES) {
		p->_m[k] = (real_t)(0);
		p->_e2[k] = (real_t)(0);
		p->_rx[k] = (real_t)(0);
		p->_ry[k] = (real_t)(0);
		p->_rz[k] = (real_t)(0);
		p->_ax[k] = (real_t)(0);
		p->_ay[k] = (real_t)(0);
		p->_az[k] = (real_t)(0);
		if (bid+k < n) {
			p->_m[k] = __m[bid+k];
			p->_e2[k] = __e2[bid+k];
			p->_rx[k] = (__rdot+(0*NDIM+0)*n)[bid+k];
			p->_ry[k] = (__rdot+(0*NDIM+1)*n)[bid+k];
			p->_rz[k] = (__rdot+(0*NDIM+2)*n)[bid+k];
		}
	}
}


static inline void
simd_shuff_Acc_Data2(uint_t lane_id, local Acc_Data *p)
{
	for (uint_t kk = 0, k = lane_id;
				kk < LMSIZE;
				kk += NLANES, k += NLANES) {
		shuff(p->m[k], SIMD);
		shuff(p->e2[k], SIMD);
		shuff(p->rx[k], SIMD);
		shuff(p->ry[k], SIMD);
		shuff(p->rz[k], SIMD);
		shuff(p->ax[k], SIMD);
		shuff(p->ay[k], SIMD);
		shuff(p->az[k], SIMD);
	}
}


static inline void
p2p_acc_kernel_core2(
	uint_t lane_id,
	local Acc_Data *ip,
	local Acc_Data *jp)
// flop count: 28
{
	for (uint_t l = 0; l < NLANES; ++l) {
		for (uint_t k = 0; k < SIMD; ++k) {
			for (uint_t ii = 0, i = lane_id;
						ii < LMSIZE;
						ii += NLANES, i += NLANES) {
				for (uint_t jj = 0, j = lane_id^l;
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
			simd_shuff_Acc_Data2(lane_id, ip);
		}
	}
}


static inline void
acc_kernel_core2(
	uint_t lane_id,
	local Acc_Data *ip,
	local Acc_Data *jp)
// flop count: 21
{
	for (uint_t l = 0; l < NLANES; ++l) {
		for (uint_t k = 0; k < SIMD; ++k) {
			for (uint_t ii = 0, i = lane_id;
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
			simd_shuff_Acc_Data2(lane_id, ip);
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
	uint_t lane_id = lid % NLANES;
	uint_t warp_id = lid / NLANES;
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	for (uint_t ii = grp;
				ii * NWARPS * LMSIZE * SIMD < ni;
				ii += ngrps) {
		uint_t ibid = (ii * NWARPS + warp_id) * LMSIZE * SIMD;
		setup_Acc_Data(
			ibid, lane_id, &ip[warp_id],
			ni, __im, __ie2, __irdot
		);

		for (uint_t jj = 0;
					jj * NWARPS * LMSIZE * SIMD < nj;
					jj += 1) {
			uint_t jbid = (jj * NWARPS + warp_id) * LMSIZE * SIMD;
			setup_Acc_Data(
				jbid, lane_id, &jp[warp_id],
				nj, __jm, __je2, __jrdot
			);

			for (uint_t w = 0; w < NWARPS; ++w) {
				p2p_acc_kernel_core2(lane_id, &ip[warp_id^w], &jp[warp_id]);
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			for (uint_t kk = 0, k = lane_id;
						kk < LMSIZE * SIMD;
						kk += NLANES, k += NLANES) {
				if (jbid+k < nj) {
					atomic_fadd(&(__jadot+(0*NDIM+0)*nj)[jbid+k], jp[warp_id]._ax[k]);
					atomic_fadd(&(__jadot+(0*NDIM+1)*nj)[jbid+k], jp[warp_id]._ay[k]);
					atomic_fadd(&(__jadot+(0*NDIM+2)*nj)[jbid+k], jp[warp_id]._az[k]);
				}
			}
		}

		for (uint_t kk = 0, k = lane_id;
					kk < LMSIZE * SIMD;
					kk += NLANES, k += NLANES) {
			if (ibid+k < ni) {
				(__iadot+(0*NDIM+0)*ni)[ibid+k] -= ip[warp_id]._ax[k];
				(__iadot+(0*NDIM+1)*ni)[ibid+k] -= ip[warp_id]._ay[k];
				(__iadot+(0*NDIM+2)*ni)[ibid+k] -= ip[warp_id]._az[k];
			}
		}
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
acc_kernel_impl2(
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
	uint_t lane_id = lid % NLANES;
	uint_t warp_id = lid / NLANES;
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	for (uint_t ii = grp;
				ii * NWARPS * LMSIZE * SIMD < ni;
				ii += ngrps) {
		uint_t ibid = (ii * NWARPS + warp_id) * LMSIZE * SIMD;
		setup_Acc_Data(
			ibid, lane_id, &ip[warp_id],
			ni, __im, __ie2, __irdot
		);

		for (uint_t jj = 0;
					jj * NWARPS * LMSIZE * SIMD < nj;
					jj += 1) {
			uint_t jbid = (jj * NWARPS + warp_id) * LMSIZE * SIMD;
			setup_Acc_Data(
				jbid, lane_id, &jp[warp_id],
				nj, __jm, __je2, __jrdot
			);

			for (uint_t w = 0; w < NWARPS; ++w) {
				acc_kernel_core2(lane_id, &ip[warp_id^w], &jp[warp_id]);
				barrier(CLK_LOCAL_MEM_FENCE);
			}

//			for (uint_t kk = 0, k = lane_id;
//						kk < LMSIZE * SIMD;
//						kk += NLANES, k += NLANES) {
//				if (jbid+k < nj) {
//					atomic_fadd(&(__jadot+(0*NDIM+0)*nj)[jbid+k], jp[warp_id]._ax[k]);
//					atomic_fadd(&(__jadot+(0*NDIM+1)*nj)[jbid+k], jp[warp_id]._ay[k]);
//					atomic_fadd(&(__jadot+(0*NDIM+2)*nj)[jbid+k], jp[warp_id]._az[k]);
//				}
//			}
		}

		for (uint_t kk = 0, k = lane_id;
					kk < LMSIZE * SIMD;
					kk += NLANES, k += NLANES) {
			if (ibid+k < ni) {
				(__iadot+(0*NDIM+0)*ni)[ibid+k] -= ip[warp_id]._ax[k];
				(__iadot+(0*NDIM+1)*ni)[ibid+k] -= ip[warp_id]._ay[k];
				(__iadot+(0*NDIM+2)*ni)[ibid+k] -= ip[warp_id]._az[k];
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
	acc_kernel_impl2(
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
	acc_kernel_impl2(
		ni, __im, __ie2, __irdot,
		nj, __jm, __je2, __jrdot,
		__iadot, __jadot
	);
	acc_kernel_impl2(
		nj, __jm, __je2, __jrdot,
		ni, __im, __ie2, __irdot,
		__jadot, __iadot
	);
}
*/

