#include "acc_jrk_kernel_common.h"


static inline void
p2p_acc_jrk_kernel_core(
	uint_t lid,
	concat(Acc_Jrk_Data, WPT) *jp,
	local concat(Acc_Jrk_Data, WGSIZE) *ip)
// flop count: 56
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
					real_tn vx = ip->vx[i] - jp->vx[j];
					real_tn vy = ip->vy[i] - jp->vy[j];
					real_tn vz = ip->vz[i] - jp->vz[j];

					real_tn rr = ee;
					rr += rx * rx;
					rr += ry * ry;
					rr += rz * rz;

					real_tn inv_r3 = rsqrt(rr);
					real_tn inv_r2 = inv_r3 * inv_r3;
					inv_r3 *= inv_r2;
					inv_r2 *= -3;

					real_tn s1  = rx * vx;
							s1 += ry * vy;
							s1 += rz * vz;

					real_tn q1 = inv_r2 * (s1);
					vx += q1 * rx;
					vy += q1 * ry;
					vz += q1 * rz;

					real_tn im_r3 = ip->m[i] * inv_r3;
					jp->ax[j] += im_r3 * rx;
					jp->ay[j] += im_r3 * ry;
					jp->az[j] += im_r3 * rz;
					jp->jx[j] += im_r3 * vx;
					jp->jy[j] += im_r3 * vy;
					jp->jz[j] += im_r3 * vz;

					real_tn jm_r3 = jp->m[j] * inv_r3;
					ip->ax[i] += jm_r3 * rx;
					ip->ay[i] += jm_r3 * ry;
					ip->az[i] += jm_r3 * rz;
					ip->jx[i] += jm_r3 * vx;
					ip->jy[i] += jm_r3 * vy;
					ip->jz[i] += jm_r3 * vz;

					simd_shuff_Acc_Jrk_Data(j, jp);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


static inline void
acc_jrk_kernel_core(
	uint_t lid,
	concat(Acc_Jrk_Data, WPT) *jp,
	local concat(Acc_Jrk_Data, WGSIZE) *ip)
// flop count: 43
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
					real_tn vx = ip->vx[i] - jp->vx[j];
					real_tn vy = ip->vy[i] - jp->vy[j];
					real_tn vz = ip->vz[i] - jp->vz[j];

					real_tn rr = ee;
					rr += rx * rx;
					rr += ry * ry;
					rr += rz * rz;

					real_tn inv_r3 = rsqrt(rr);
					inv_r3 = (rr > ee) ? (inv_r3):(0);
					real_tn inv_r2 = inv_r3 * inv_r3;
					inv_r3 *= inv_r2;
					inv_r2 *= -3;

					real_tn s1  = rx * vx;
							s1 += ry * vy;
							s1 += rz * vz;

					real_tn q1 = inv_r2 * (s1);
					vx += q1 * rx;
					vy += q1 * ry;
					vz += q1 * rz;

					real_tn im_r3 = ip->m[i] * inv_r3;
					jp->ax[j] += im_r3 * rx;
					jp->ay[j] += im_r3 * ry;
					jp->az[j] += im_r3 * rz;
					jp->jx[j] += im_r3 * vx;
					jp->jy[j] += im_r3 * vy;
					jp->jz[j] += im_r3 * vz;

//					real_tn jm_r3 = jp->m[j] * inv_r3;
//					ip->ax[i] += jm_r3 * rx;
//					ip->ay[i] += jm_r3 * ry;
//					ip->az[i] += jm_r3 * rz;
//					ip->jx[i] += jm_r3 * vx;
//					ip->jy[i] += jm_r3 * vy;
//					ip->jz[i] += jm_r3 * vz;

					simd_shuff_Acc_Jrk_Data(j, jp);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
acc_jrk_kernel_rectangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	global real_t __iadot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	global real_t __jadot[])
{
	local concat(Acc_Jrk_Data, WGSIZE) _ip;
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t block = WGSIZE * SIMD * WPT;
	for (uint_t jj = block * grp;
				jj < nj;
				jj += block * ngrps) {
		concat(Acc_Jrk_Data, WPT) jp = {{{0}}};
		concat(load_Acc_Jrk_Data, WPT)(
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
				concat(Acc_Jrk_Data, 1) ip = {{{0}}};
				concat(load_Acc_Jrk_Data, 1)(
					&ip, ii + ilid, WGSIZE, SIMD,
					ni, __im, __ie2, __irdot
				);
				_ip.m[lid] = ip.m[0];
				_ip.e2[lid] = ip.e2[0];
				_ip.rx[lid] = ip.rx[0];
				_ip.ry[lid] = ip.ry[0];
				_ip.rz[lid] = ip.rz[0];
				_ip.vx[lid] = ip.vx[0];
				_ip.vy[lid] = ip.vy[0];
				_ip.vz[lid] = ip.vz[0];
				_ip.ax[lid] = ip.ax[0];
				_ip.ay[lid] = ip.ay[0];
				_ip.az[lid] = ip.az[0];
				_ip.jx[lid] = ip.jx[0];
				_ip.jy[lid] = ip.jy[0];
				_ip.jz[lid] = ip.jz[0];

				if (ii != jj) {
					p2p_acc_jrk_kernel_core(lid, &jp, &_ip);
				} else {
					p2p_acc_jrk_kernel_core(lid, &jp, &_ip);
				}

				ip.ax[0] = -_ip.ax[lid];
				ip.ay[0] = -_ip.ay[lid];
				ip.az[0] = -_ip.az[lid];
				ip.jx[0] = -_ip.jx[lid];
				ip.jy[0] = -_ip.jy[lid];
				ip.jz[0] = -_ip.jz[lid];
				concat(store_Acc_Jrk_Data, 1)(
					&ip, ii + ilid, WGSIZE, SIMD,
					ni, __iadot
				);
			}
		}

		concat(store_Acc_Jrk_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jadot
		);
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
acc_jrk_kernel_triangle(
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

	local concat(Acc_Jrk_Data, WGSIZE) _ip;
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t block = WGSIZE * SIMD * WPT;
	for (uint_t jj = block * grp;
				jj < nj;
				jj += block * ngrps) {
		concat(Acc_Jrk_Data, WPT) jp = {{{0}}};
		concat(load_Acc_Jrk_Data, WPT)(
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
				concat(Acc_Jrk_Data, 1) ip = {{{0}}};
				concat(load_Acc_Jrk_Data, 1)(
					&ip, ii + ilid, WGSIZE, SIMD,
					ni, __im, __ie2, __irdot
				);
				_ip.m[lid] = ip.m[0];
				_ip.e2[lid] = ip.e2[0];
				_ip.rx[lid] = ip.rx[0];
				_ip.ry[lid] = ip.ry[0];
				_ip.rz[lid] = ip.rz[0];
				_ip.vx[lid] = ip.vx[0];
				_ip.vy[lid] = ip.vy[0];
				_ip.vz[lid] = ip.vz[0];
				_ip.ax[lid] = ip.ax[0];
				_ip.ay[lid] = ip.ay[0];
				_ip.az[lid] = ip.az[0];
				_ip.jx[lid] = ip.jx[0];
				_ip.jy[lid] = ip.jy[0];
				_ip.jz[lid] = ip.jz[0];

				if (ii != jj) {
					p2p_acc_jrk_kernel_core(lid, &jp, &_ip);
				} else {
					acc_jrk_kernel_core(lid, &jp, &_ip);
				}

				ip.ax[0] = -_ip.ax[lid];
				ip.ay[0] = -_ip.ay[lid];
				ip.az[0] = -_ip.az[lid];
				ip.jx[0] = -_ip.jx[lid];
				ip.jy[0] = -_ip.jy[lid];
				ip.jz[0] = -_ip.jz[lid];
				concat(store_Acc_Jrk_Data, 1)(
					&ip, ii + ilid, WGSIZE, SIMD,
					ni, __iadot
				);
			}
		}

		concat(store_Acc_Jrk_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jadot
		);
	}
}

