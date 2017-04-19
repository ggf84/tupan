#include "phi_kernel_common.h"


static inline void
p2p_phi_kernel_core(
	uint_t lane,
	Phi_Data *jp,
	local Phi_Data_SoA *ip)
// flop count: 16
{
	#pragma unroll
	for (uint_t l = 0; l < NLANES; ++l) {
		uint_t i = lane^l;
		#pragma unroll
		for (uint_t k = 0; k < SIMD; ++k) {
			real_tn ee = ip->e2[i] + jp->e2;
			real_tn rx = ip->rx[i] - jp->rx;
			real_tn ry = ip->ry[i] - jp->ry;
			real_tn rz = ip->rz[i] - jp->rz;

			real_tn rr = ee;
			rr += rx * rx;
			rr += ry * ry;
			rr += rz * rz;

			real_tn inv_r1 = rsqrt(rr);

			jp->phi += ip->m[i] * inv_r1;
			ip->phi[i] += jp->m * inv_r1;

			simd_shuff_Phi_Data(jp);
		}
	}
}


static inline void
phi_kernel_core(
	uint_t lane,
	Phi_Data *jp,
	local Phi_Data_SoA *ip)
// flop count: 14
{
	#pragma unroll
	for (uint_t l = 0; l < NLANES; ++l) {
		uint_t i = lane^l;
		#pragma unroll
		for (uint_t k = 0; k < SIMD; ++k) {
			real_tn ee = ip->e2[i] + jp->e2;
			real_tn rx = ip->rx[i] - jp->rx;
			real_tn ry = ip->ry[i] - jp->ry;
			real_tn rz = ip->rz[i] - jp->rz;

			real_tn rr = ee;
			rr += rx * rx;
			rr += ry * ry;
			rr += rz * rz;

			real_tn inv_r1 = rsqrt(rr);
			inv_r1 = (rr > ee) ? (inv_r1):(0);

//			jp->phi += ip->m[i] * inv_r1;
			ip->phi[i] += jp->m * inv_r1;

			simd_shuff_Phi_Data(jp);
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
	local Phi_Data_SoA _ip[NWARPS];
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t lane = lid % NLANES;
	uint_t warp = lid / NLANES;
	for (uint_t ii = SIMD * WGSIZE * grp;
				ii < ni;
				ii += SIMD * WGSIZE * ngrps) {
		Phi_Data ip = {{0}};
		read_Phi_Data(
			ii, lid, &ip,
			ni, __im, __ie2, __irdot
		);
		_ip[warp].m[lane] = ip.m;
		_ip[warp].e2[lane] = ip.e2;
		_ip[warp].rx[lane] = ip.rx;
		_ip[warp].ry[lane] = ip.ry;
		_ip[warp].rz[lane] = ip.rz;
		_ip[warp].phi[lane] = ip.phi;

		for (uint_t jj = 0;
					jj < nj;
					jj += SIMD * WGSIZE) {
			Phi_Data jp = {{0}};
			read_Phi_Data(
				jj, lid, &jp,
				nj, __jm, __je2, __jrdot
			);

			for (uint_t w = 0; w < NWARPS; ++w) {
				p2p_phi_kernel_core(lane, &jp, &_ip[(warp+w)%NWARPS]);
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			for (uint_t k = 0, kk = jj + lid;
						k < SIMD;
						k += 1, kk += WGSIZE) {
				if (kk < nj) {
					atomic_fadd(&__jphi[kk], -jp._phi[k]);
				}
			}
		}

		ip.phi = _ip[warp].phi[lane];
		for (uint_t k = 0, kk = ii + lid;
					k < SIMD;
					k += 1, kk += WGSIZE) {
			if (kk < ni) {
				__iphi[kk] -= ip._phi[k];
			}
		}
	}
}


kernel __attribute__((reqd_work_group_size(WGSIZE, 1, 1))) void
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

	local Phi_Data_SoA _ip[NWARPS];
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t lane = lid % NLANES;
	uint_t warp = lid / NLANES;
	for (uint_t ii = SIMD * WGSIZE * grp;
				ii < ni;
				ii += SIMD * WGSIZE * ngrps) {
		Phi_Data ip = {{0}};
		read_Phi_Data(
			ii, lid, &ip,
			ni, __im, __ie2, __irdot
		);
		_ip[warp].m[lane] = ip.m;
		_ip[warp].e2[lane] = ip.e2;
		_ip[warp].rx[lane] = ip.rx;
		_ip[warp].ry[lane] = ip.ry;
		_ip[warp].rz[lane] = ip.rz;
		_ip[warp].phi[lane] = ip.phi;

		for (uint_t jj = 0;
					jj < nj;
					jj += SIMD * WGSIZE) {
			Phi_Data jp = {{0}};
			read_Phi_Data(
				jj, lid, &jp,
				nj, __jm, __je2, __jrdot
			);

			for (uint_t w = 0; w < NWARPS; ++w) {
				phi_kernel_core(lane, &jp, &_ip[(warp+w)%NWARPS]);
				barrier(CLK_LOCAL_MEM_FENCE);
			}

//			for (uint_t k = 0, kk = jj + lid;
//						k < SIMD;
//						k += 1, kk += WGSIZE) {
//				if (kk < nj) {
//					atomic_fadd(&__jphi[kk], -jp._phi[k]);
//				}
//			}
		}

		ip.phi = _ip[warp].phi[lane];
		for (uint_t k = 0, kk = ii + lid;
					k < SIMD;
					k += 1, kk += WGSIZE) {
			if (kk < ni) {
				__iphi[kk] -= ip._phi[k];
			}
		}
	}
}

