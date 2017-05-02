#include "acc_jrk_kernel_common.h"


static inline void
p2p_acc_jrk_kernel_core(
	uint_t lane,
	concat(Acc_Jrk_Data, WPT) *jp,
	local concat(Acc_Jrk_Data, NLANES) *ip)
// flop count: 56
{
	for (uint_t l = 0; l < NLANES; ++l) {
		uint_t i = lane^l;
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
}


static inline void
acc_jrk_kernel_core(
	uint_t lane,
	concat(Acc_Jrk_Data, WPT) *jp,
	local concat(Acc_Jrk_Data, NLANES) *ip)
// flop count: 43
{
	for (uint_t l = 0; l < NLANES; ++l) {
		uint_t i = lane^l;
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

//				real_tn jm_r3 = jp->m[j] * inv_r3;
//				ip->ax[i] += jm_r3 * rx;
//				ip->ay[i] += jm_r3 * ry;
//				ip->az[i] += jm_r3 * rz;
//				ip->jx[i] += jm_r3 * vx;
//				ip->jy[i] += jm_r3 * vy;
//				ip->jz[i] += jm_r3 * vz;

				simd_shuff_Acc_Jrk_Data(j, jp);
			}
		}
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
acc_jrk_kernel_rectangle(
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
	local concat(Acc_Jrk_Data, NLANES) _ip[NWARPS];
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t lane = lid % NLANES;
	uint_t warp = lid / NLANES;
	for (uint_t jj = 0;
				jj < nj;
				jj += WGSIZE * SIMD * WPT) {
		concat(Acc_Jrk_Data, WPT) jp = {{{0}}};
		concat(read_Acc_Jrk_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jm, __je2, __jrdot
		);

		for (uint_t ii = WGSIZE * SIMD * grp;
					ii < ni;
					ii += WGSIZE * SIMD * ngrps) {
			concat(Acc_Jrk_Data, 1) ip = {{{0}}};
			concat(read_Acc_Jrk_Data, 1)(
				&ip, ii + lid, WGSIZE, SIMD,
				ni, __im, __ie2, __irdot
			);
			_ip[warp].m[lane] = ip.m[0];
			_ip[warp].e2[lane] = ip.e2[0];
			_ip[warp].rx[lane] = ip.rx[0];
			_ip[warp].ry[lane] = ip.ry[0];
			_ip[warp].rz[lane] = ip.rz[0];
			_ip[warp].vx[lane] = ip.vx[0];
			_ip[warp].vy[lane] = ip.vy[0];
			_ip[warp].vz[lane] = ip.vz[0];
			_ip[warp].ax[lane] = ip.ax[0];
			_ip[warp].ay[lane] = ip.ay[0];
			_ip[warp].az[lane] = ip.az[0];
			_ip[warp].jx[lane] = ip.jx[0];
			_ip[warp].jy[lane] = ip.jy[0];
			_ip[warp].jz[lane] = ip.jz[0];

			for (uint_t w = 0; w < NWARPS; ++w) {
				p2p_acc_jrk_kernel_core(lane, &jp, &_ip[(warp+w)%NWARPS]);
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			ip.ax[0] = _ip[warp].ax[lane];
			ip.ay[0] = _ip[warp].ay[lane];
			ip.az[0] = _ip[warp].az[lane];
			ip.jx[0] = _ip[warp].jx[lane];
			ip.jy[0] = _ip[warp].jy[lane];
			ip.jz[0] = _ip[warp].jz[lane];
			for (uint_t k = 0, kk = ii + lid;
						k < SIMD;
						k += 1, kk += WGSIZE) {
				if (kk < ni) {
					(__iadot+(0*NDIM+0)*ni)[kk] -= ip._ax[k];
					(__iadot+(0*NDIM+1)*ni)[kk] -= ip._ay[k];
					(__iadot+(0*NDIM+2)*ni)[kk] -= ip._az[k];
					(__iadot+(1*NDIM+0)*ni)[kk] -= ip._jx[k];
					(__iadot+(1*NDIM+1)*ni)[kk] -= ip._jy[k];
					(__iadot+(1*NDIM+2)*ni)[kk] -= ip._jz[k];
				}
			}
		}

		for (uint_t k = 0, kk = jj + lid;
					k < SIMD * WPT;
					k += 1, kk += WGSIZE) {
			if (kk < nj) {
				atomic_fadd(&(__jadot+(0*NDIM+0)*nj)[kk], jp._ax[k]);
				atomic_fadd(&(__jadot+(0*NDIM+1)*nj)[kk], jp._ay[k]);
				atomic_fadd(&(__jadot+(0*NDIM+2)*nj)[kk], jp._az[k]);
				atomic_fadd(&(__jadot+(1*NDIM+0)*nj)[kk], jp._jx[k]);
				atomic_fadd(&(__jadot+(1*NDIM+1)*nj)[kk], jp._jy[k]);
				atomic_fadd(&(__jadot+(1*NDIM+2)*nj)[kk], jp._jz[k]);
			}
		}
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

	local concat(Acc_Jrk_Data, NLANES) _ip[NWARPS];
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t lane = lid % NLANES;
	uint_t warp = lid / NLANES;
	for (uint_t jj = 0;
				jj < nj;
				jj += WGSIZE * SIMD * WPT) {
		concat(Acc_Jrk_Data, WPT) jp = {{{0}}};
		concat(read_Acc_Jrk_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jm, __je2, __jrdot
		);

		for (uint_t ii = WGSIZE * SIMD * grp;
					ii < ni;
					ii += WGSIZE * SIMD * ngrps) {
			concat(Acc_Jrk_Data, 1) ip = {{{0}}};
			concat(read_Acc_Jrk_Data, 1)(
				&ip, ii + lid, WGSIZE, SIMD,
				ni, __im, __ie2, __irdot
			);
			_ip[warp].m[lane] = ip.m[0];
			_ip[warp].e2[lane] = ip.e2[0];
			_ip[warp].rx[lane] = ip.rx[0];
			_ip[warp].ry[lane] = ip.ry[0];
			_ip[warp].rz[lane] = ip.rz[0];
			_ip[warp].vx[lane] = ip.vx[0];
			_ip[warp].vy[lane] = ip.vy[0];
			_ip[warp].vz[lane] = ip.vz[0];
			_ip[warp].ax[lane] = ip.ax[0];
			_ip[warp].ay[lane] = ip.ay[0];
			_ip[warp].az[lane] = ip.az[0];
			_ip[warp].jx[lane] = ip.jx[0];
			_ip[warp].jy[lane] = ip.jy[0];
			_ip[warp].jz[lane] = ip.jz[0];

			for (uint_t w = 0; w < NWARPS; ++w) {
				acc_jrk_kernel_core(lane, &jp, &_ip[(warp+w)%NWARPS]);
				barrier(CLK_LOCAL_MEM_FENCE);
			}

//			ip.ax[0] = _ip[warp].ax[lane];
//			ip.ay[0] = _ip[warp].ay[lane];
//			ip.az[0] = _ip[warp].az[lane];
//			ip.jx[0] = _ip[warp].jx[lane];
//			ip.jy[0] = _ip[warp].jy[lane];
//			ip.jz[0] = _ip[warp].jz[lane];
//			for (uint_t k = 0, kk = ii + lid;
//						k < SIMD;
//						k += 1, kk += WGSIZE) {
//				if (kk < ni) {
//					(__iadot+(0*NDIM+0)*ni)[kk] -= ip._ax[k];
//					(__iadot+(0*NDIM+1)*ni)[kk] -= ip._ay[k];
//					(__iadot+(0*NDIM+2)*ni)[kk] -= ip._az[k];
//					(__iadot+(1*NDIM+0)*ni)[kk] -= ip._jx[k];
//					(__iadot+(1*NDIM+1)*ni)[kk] -= ip._jy[k];
//					(__iadot+(1*NDIM+2)*ni)[kk] -= ip._jz[k];
//				}
//			}
		}

		for (uint_t k = 0, kk = jj + lid;
					k < SIMD * WPT;
					k += 1, kk += WGSIZE) {
			if (kk < nj) {
				atomic_fadd(&(__jadot+(0*NDIM+0)*nj)[kk], jp._ax[k]);
				atomic_fadd(&(__jadot+(0*NDIM+1)*nj)[kk], jp._ay[k]);
				atomic_fadd(&(__jadot+(0*NDIM+2)*nj)[kk], jp._az[k]);
				atomic_fadd(&(__jadot+(1*NDIM+0)*nj)[kk], jp._jx[k]);
				atomic_fadd(&(__jadot+(1*NDIM+1)*nj)[kk], jp._jy[k]);
				atomic_fadd(&(__jadot+(1*NDIM+2)*nj)[kk], jp._jz[k]);
			}
		}
	}
}

