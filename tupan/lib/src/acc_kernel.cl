#include "acc_kernel_common.h"


static inline void
p2p_acc_kernel_core(
	uint_t lane,
	concat(Acc_Data, WPT) *jp,
	local concat(Acc_Data, NLANES) *ip)
// flop count: 28
{
	for (uint_t l = 0; l < NLANES; ++l) {
		uint_t i = lane^l;
		for (uint_t j = 0; j < WPT; ++j) {
			for (uint_t k = 0; k < SIMD; ++k) {
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

				simd_shuff_Acc_Data(j, jp);
			}
		}
	}
}


static inline void
acc_kernel_core(
	uint_t lane,
	concat(Acc_Data, WPT) *jp,
	local concat(Acc_Data, NLANES) *ip)
// flop count: 21
{
	for (uint_t l = 0; l < NLANES; ++l) {
		uint_t i = lane^l;
		for (uint_t j = 0; j < WPT; ++j) {
			for (uint_t k = 0; k < SIMD; ++k) {
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

				real_tn im_r3 = ip->m[i] * inv_r3;
				jp->ax[j] += im_r3 * rx;
				jp->ay[j] += im_r3 * ry;
				jp->az[j] += im_r3 * rz;

//				real_tn jm_r3 = jp->m[j] * inv_r3;
//				ip->ax[i] += jm_r3 * rx;
//				ip->ay[i] += jm_r3 * ry;
//				ip->az[i] += jm_r3 * rz;

				simd_shuff_Acc_Data(j, jp);
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
	local concat(Acc_Data, NLANES) _ip[NWARPS];
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t lane = lid % NLANES;
	uint_t warp = lid / NLANES;
	uint_t block = WGSIZE * SIMD;
	for (uint_t jj = WPT * block * grp;
				jj < nj;
				jj += WPT * block * ngrps) {
		concat(Acc_Data, WPT) jp = {{{0}}};
		concat(load_Acc_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jm, __je2, __jrdot
		);

		for (uint_t b = 0; b < WPT; ++b)
		for (uint_t ii = b * block;
					ii < ni;
					ii += WPT * block) {
			concat(Acc_Data, 1) ip = {{{0}}};
			concat(load_Acc_Data, 1)(
				&ip, ii + lid, WGSIZE, SIMD,
				ni, __im, __ie2, __irdot
			);
			_ip[warp].m[lane] = ip.m[0];
			_ip[warp].e2[lane] = ip.e2[0];
			_ip[warp].rx[lane] = ip.rx[0];
			_ip[warp].ry[lane] = ip.ry[0];
			_ip[warp].rz[lane] = ip.rz[0];
			_ip[warp].ax[lane] = ip.ax[0];
			_ip[warp].ay[lane] = ip.ay[0];
			_ip[warp].az[lane] = ip.az[0];

			uint_t i = ii / (WPT * block);
			uint_t j = jj / (WPT * block);
			if (i == j) {
				for (uint_t w = 0; w < NWARPS; ++w) {
					p2p_acc_kernel_core(lane, &jp, &_ip[(warp+w)%NWARPS]);
					barrier(CLK_LOCAL_MEM_FENCE);
				}
			} else if ((i > j/* && (i + j) % 2 == 0*/)
					|| (i < j/* && (i + j) % 2 == 1*/)) {
				for (uint_t w = 0; w < NWARPS; ++w) {
					p2p_acc_kernel_core(lane, &jp, &_ip[(warp+w)%NWARPS]);
					barrier(CLK_LOCAL_MEM_FENCE);
				}
			}

			ip.ax[0] = -_ip[warp].ax[lane];
			ip.ay[0] = -_ip[warp].ay[lane];
			ip.az[0] = -_ip[warp].az[lane];
			concat(store_Acc_Data, 1)(
				&ip, ii + lid, WGSIZE, SIMD,
				ni, __iadot
			);
		}

		concat(store_Acc_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jadot
		);
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

	local concat(Acc_Data, NLANES) _ip[NWARPS];
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t lane = lid % NLANES;
	uint_t warp = lid / NLANES;
	uint_t block = WGSIZE * SIMD;
	for (uint_t jj = WPT * block * grp;
				jj < nj;
				jj += WPT * block * ngrps) {
		concat(Acc_Data, WPT) jp = {{{0}}};
		concat(load_Acc_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jm, __je2, __jrdot
		);

		for (uint_t b = 0; b < WPT; ++b)
		for (uint_t ii = b * block;
					ii < ni;
					ii += WPT * block) {
			concat(Acc_Data, 1) ip = {{{0}}};
			concat(load_Acc_Data, 1)(
				&ip, ii + lid, WGSIZE, SIMD,
				ni, __im, __ie2, __irdot
			);
			_ip[warp].m[lane] = ip.m[0];
			_ip[warp].e2[lane] = ip.e2[0];
			_ip[warp].rx[lane] = ip.rx[0];
			_ip[warp].ry[lane] = ip.ry[0];
			_ip[warp].rz[lane] = ip.rz[0];
			_ip[warp].ax[lane] = ip.ax[0];
			_ip[warp].ay[lane] = ip.ay[0];
			_ip[warp].az[lane] = ip.az[0];

			uint_t i = ii / (WPT * block);
			uint_t j = jj / (WPT * block);
			if (i == j) {
				for (uint_t w = 0; w < NWARPS; ++w) {
					acc_kernel_core(lane, &jp, &_ip[(warp+w)%NWARPS]);
					barrier(CLK_LOCAL_MEM_FENCE);
				}
			} else if ((i > j && (i + j) % 2 == 0)
					|| (i < j && (i + j) % 2 == 1)) {
				for (uint_t w = 0; w < NWARPS; ++w) {
					p2p_acc_kernel_core(lane, &jp, &_ip[(warp+w)%NWARPS]);
					barrier(CLK_LOCAL_MEM_FENCE);
				}
			}

			ip.ax[0] = -_ip[warp].ax[lane];
			ip.ay[0] = -_ip[warp].ay[lane];
			ip.az[0] = -_ip[warp].az[lane];
			concat(store_Acc_Data, 1)(
				&ip, ii + lid, WGSIZE, SIMD,
				ni, __iadot
			);
		}

		concat(store_Acc_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jadot
		);
	}
}

