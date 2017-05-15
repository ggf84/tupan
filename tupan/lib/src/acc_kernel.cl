#include "acc_kernel_common.h"


static inline void
p2p_acc_kernel_core(
	uint_t lid,
	concat(Acc_Data, WPT) *jp,
	local concat(Acc_Data, WGSIZE) *ip)
// flop count: 28
{
	#pragma unroll
	for (uint_t w = 0; w < WGSIZE; w += NLANES) {
		for (uint_t l = w; l < w + NLANES; ++l) {
			uint_t i = lid^l;
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
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


static inline void
acc_kernel_core(
	uint_t lid,
	concat(Acc_Data, WPT) *jp,
	local concat(Acc_Data, WGSIZE) *ip)
// flop count: 21
{
	#pragma unroll
	for (uint_t w = 0; w < WGSIZE; w += NLANES) {
		for (uint_t l = w; l < w + NLANES; ++l) {
			uint_t i = lid^l;
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

//					real_tn jm_r3 = jp->m[j] * inv_r3;
//					ip->ax[i] += jm_r3 * rx;
//					ip->ay[i] += jm_r3 * ry;
//					ip->az[i] += jm_r3 * rz;

					simd_shuff_Acc_Data(j, jp);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
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
	local concat(Acc_Data, WGSIZE) _ip;
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t block = WGSIZE * SIMD * WPT;
	for (uint_t jj = block * grp;
				jj < nj;
				jj += block * ngrps) {
		concat(Acc_Data, WPT) jp = {{{0}}};
		concat(load_Acc_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jm, __je2, __jrdot
		);

		for (uint_t ii = 0;
					ii < ni;
					ii += block)
			if ((ii == jj)
				|| (ii > jj/* && ((ii + jj) / block) % 2 == 0*/)
				|| (ii < jj/* && ((ii + jj) / block) % 2 == 1*/)) {
			for (uint_t ilid = lid;
						ilid < block;
						ilid += WGSIZE * SIMD) {
				concat(Acc_Data, 1) ip = {{{0}}};
				concat(load_Acc_Data, 1)(
					&ip, ii + ilid, WGSIZE, SIMD,
					ni, __im, __ie2, __irdot
				);
				_ip.m[lid] = ip.m[0];
				_ip.e2[lid] = ip.e2[0];
				_ip.rx[lid] = ip.rx[0];
				_ip.ry[lid] = ip.ry[0];
				_ip.rz[lid] = ip.rz[0];
				_ip.ax[lid] = ip.ax[0];
				_ip.ay[lid] = ip.ay[0];
				_ip.az[lid] = ip.az[0];

				if (ii != jj) {
					p2p_acc_kernel_core(lid, &jp, &_ip);
				} else {
					p2p_acc_kernel_core(lid, &jp, &_ip);
				}

				ip.ax[0] = -_ip.ax[lid];
				ip.ay[0] = -_ip.ay[lid];
				ip.az[0] = -_ip.az[lid];
				concat(store_Acc_Data, 1)(
					&ip, ii + ilid, WGSIZE, SIMD,
					ni, __iadot
				);
			}
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

	local concat(Acc_Data, WGSIZE) _ip;
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t block = WGSIZE * SIMD * WPT;
	for (uint_t jj = block * grp;
				jj < nj;
				jj += block * ngrps) {
		concat(Acc_Data, WPT) jp = {{{0}}};
		concat(load_Acc_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jm, __je2, __jrdot
		);

		for (uint_t ii = 0;
					ii < ni;
					ii += block)
			if ((ii == jj)
				|| (ii > jj && ((ii + jj) / block) % 2 == 0)
				|| (ii < jj && ((ii + jj) / block) % 2 == 1)) {
			for (uint_t ilid = lid;
						ilid < block;
						ilid += WGSIZE * SIMD) {
				concat(Acc_Data, 1) ip = {{{0}}};
				concat(load_Acc_Data, 1)(
					&ip, ii + ilid, WGSIZE, SIMD,
					ni, __im, __ie2, __irdot
				);
				_ip.m[lid] = ip.m[0];
				_ip.e2[lid] = ip.e2[0];
				_ip.rx[lid] = ip.rx[0];
				_ip.ry[lid] = ip.ry[0];
				_ip.rz[lid] = ip.rz[0];
				_ip.ax[lid] = ip.ax[0];
				_ip.ay[lid] = ip.ay[0];
				_ip.az[lid] = ip.az[0];

				if (ii != jj) {
					p2p_acc_kernel_core(lid, &jp, &_ip);
				} else {
					acc_kernel_core(lid, &jp, &_ip);
				}

				ip.ax[0] = -_ip.ax[lid];
				ip.ay[0] = -_ip.ay[lid];
				ip.az[0] = -_ip.az[lid];
				concat(store_Acc_Data, 1)(
					&ip, ii + ilid, WGSIZE, SIMD,
					ni, __iadot
				);
			}
		}

		concat(store_Acc_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jadot
		);
	}
}

