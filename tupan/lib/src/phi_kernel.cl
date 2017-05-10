#include "phi_kernel_common.h"


static inline void
p2p_phi_kernel_core(
	uint_t lane,
	concat(Phi_Data, WPT) *jp,
	local concat(Phi_Data, NLANES) *ip)
// flop count: 16
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

				real_tn inv_r1 = rsqrt(rr);

				jp->phi[j] += ip->m[i] * inv_r1;
				ip->phi[i] += jp->m[j] * inv_r1;

				simd_shuff_Phi_Data(j, jp);
			}
		}
	}
}


static inline void
phi_kernel_core(
	uint_t lane,
	concat(Phi_Data, WPT) *jp,
	local concat(Phi_Data, NLANES) *ip)
// flop count: 14
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

				real_tn inv_r1 = rsqrt(rr);
				inv_r1 = (rr > ee) ? (inv_r1):(0);

				jp->phi[j] += ip->m[i] * inv_r1;
//				ip->phi[i] += jp->m[j] * inv_r1;

				simd_shuff_Phi_Data(j, jp);
			}
		}
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
phi_kernel_rectangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	global real_t __iphi[],
	global real_t __jphi[])
{
	local concat(Phi_Data, NLANES) _ip[NWARPS];
	uint_t lid = get_local_id(0);
	uint_t grp = get_global_id(0) / NLANES;
	uint_t ngrps = get_global_size(0) / NLANES;
	uint_t lane = lid % NLANES;
	uint_t warp = lid / NLANES;
	uint_t block = NLANES * SIMD * WPT;
	for (uint_t jj = block * grp;
				jj < nj;
				jj += block * ngrps) {
		concat(Phi_Data, WPT) jp = {{{0}}};
		concat(load_Phi_Data, WPT)(
			&jp, jj + lane, NLANES, SIMD,
			nj, __jm, __je2, __jrdot
		);

		for (uint_t ilane = lane;
					ilane < block;
					ilane += NLANES * SIMD)
		for (uint_t ii = 0;
					ii < ni;
					ii += block) {
			if ((ii == jj)
				|| (ii > jj/* && ((ii + jj) / block) % 2 == 0*/)
				|| (ii < jj/* && ((ii + jj) / block) % 2 == 1*/)) {
				concat(Phi_Data, 1) ip = {{{0}}};
				concat(load_Phi_Data, 1)(
					&ip, ii + ilane, NLANES, SIMD,
					ni, __im, __ie2, __irdot
				);
				_ip[warp].m[lane] = ip.m[0];
				_ip[warp].e2[lane] = ip.e2[0];
				_ip[warp].rx[lane] = ip.rx[0];
				_ip[warp].ry[lane] = ip.ry[0];
				_ip[warp].rz[lane] = ip.rz[0];
				_ip[warp].phi[lane] = ip.phi[0];

				if (ii != jj) {
					p2p_phi_kernel_core(lane, &jp, &_ip[warp]);
				} else {
					p2p_phi_kernel_core(lane, &jp, &_ip[warp]);
				}

				ip.phi[0] = _ip[warp].phi[lane];
				concat(store_Phi_Data, 1)(
					&ip, ii + ilane, NLANES, SIMD,
					ni, __iphi
				);
			}
		}

		concat(store_Phi_Data, WPT)(
			&jp, jj + lane, NLANES, SIMD,
			nj, __jphi
		);
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
phi_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	global real_t __iphi[])
{
	// ------------------------------------------------------------------------
	const uint_t nj = ni;
	global const real_t *__jm = __im;
	global const real_t *__je2 = __ie2;
	global const real_t *__jrdot = __irdot;
	global real_t *__jphi = __iphi;
	// ------------------------------------------------------------------------

	local concat(Phi_Data, NLANES) _ip[NWARPS];
	uint_t lid = get_local_id(0);
	uint_t grp = get_global_id(0) / NLANES;
	uint_t ngrps = get_global_size(0) / NLANES;
	uint_t lane = lid % NLANES;
	uint_t warp = lid / NLANES;
	uint_t block = NLANES * SIMD * WPT;
	for (uint_t jj = block * grp;
				jj < nj;
				jj += block * ngrps) {
		concat(Phi_Data, WPT) jp = {{{0}}};
		concat(load_Phi_Data, WPT)(
			&jp, jj + lane, NLANES, SIMD,
			nj, __jm, __je2, __jrdot
		);

		for (uint_t ilane = lane;
					ilane < block;
					ilane += NLANES * SIMD)
		for (uint_t ii = 0;
					ii < ni;
					ii += block) {
			if ((ii == jj)
				|| (ii > jj && ((ii + jj) / block) % 2 == 0)
				|| (ii < jj && ((ii + jj) / block) % 2 == 1)) {
				concat(Phi_Data, 1) ip = {{{0}}};
				concat(load_Phi_Data, 1)(
					&ip, ii + ilane, NLANES, SIMD,
					ni, __im, __ie2, __irdot
				);
				_ip[warp].m[lane] = ip.m[0];
				_ip[warp].e2[lane] = ip.e2[0];
				_ip[warp].rx[lane] = ip.rx[0];
				_ip[warp].ry[lane] = ip.ry[0];
				_ip[warp].rz[lane] = ip.rz[0];
				_ip[warp].phi[lane] = ip.phi[0];

				if (ii != jj) {
					p2p_phi_kernel_core(lane, &jp, &_ip[warp]);
				} else {
					phi_kernel_core(lane, &jp, &_ip[warp]);
				}

				ip.phi[0] = _ip[warp].phi[lane];
				concat(store_Phi_Data, 1)(
					&ip, ii + ilane, NLANES, SIMD,
					ni, __iphi
				);
			}
		}

		concat(store_Phi_Data, WPT)(
			&jp, jj + lane, NLANES, SIMD,
			nj, __jphi
		);
	}
}

