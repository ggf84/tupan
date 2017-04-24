#include "tstep_kernel_common.h"


static inline void
p2p_tstep_kernel_core(
	const real_t eta,
	uint_t lane,
	Tstep_Data *jp,
	local Tstep_Data_SoA *ip)
// flop count: 43
{
	#pragma unroll
	for (uint_t l = 0; l < NLANES; ++l) {
		uint_t i = lane^l;
		#pragma unroll
		for (uint_t k = 0; k < SIMD; ++k) {
			real_tn m_r3 = ip->m[i] + jp->m;
			real_tn ee = ip->e2[i] + jp->e2;
			real_tn rx = ip->rx[i] - jp->rx;
			real_tn ry = ip->ry[i] - jp->ry;
			real_tn rz = ip->rz[i] - jp->rz;
			real_tn vx = ip->vx[i] - jp->vx;
			real_tn vy = ip->vy[i] - jp->vy;
			real_tn vz = ip->vz[i] - jp->vz;

			real_tn rr = ee;
			rr += rx * rx;
			rr += ry * ry;
			rr += rz * rz;

			real_tn rv = rx * vx;
			rv += ry * vy;
			rv += rz * vz;

			real_tn vv = vx * vx;
			vv += vy * vy;
			vv += vz * vz;

			real_tn inv_r2 = rsqrt(rr);
			m_r3 *= inv_r2;
			inv_r2 *= inv_r2;
			m_r3 *= 2 * inv_r2;

			real_tn m_r5 = m_r3 * inv_r2;
			m_r3 += vv * inv_r2;
			rv *= eta * rsqrt(m_r3);
			m_r5 += m_r3 * inv_r2;
			m_r3 -= m_r5 * rv;

			m_r3 = (ip->m[i] == 0 || jp->m == 0) ? (0):(m_r3);

			jp->w2_a = fmax(m_r3, jp->w2_a);
			jp->w2_b += m_r3;

			ip->w2_a[i] = fmax(m_r3, ip->w2_a[i]);
			ip->w2_b[i] += m_r3;

			simd_shuff_Tstep_Data(jp);
		}
	}
}


static inline void
tstep_kernel_core(
	const real_t eta,
	uint_t lane,
	Tstep_Data *jp,
	local Tstep_Data_SoA *ip)
// flop count: 42
{
	#pragma unroll
	for (uint_t l = 0; l < NLANES; ++l) {
		uint_t i = lane^l;
		#pragma unroll
		for (uint_t k = 0; k < SIMD; ++k) {
			real_tn m_r3 = ip->m[i] + jp->m;
			real_tn ee = ip->e2[i] + jp->e2;
			real_tn rx = ip->rx[i] - jp->rx;
			real_tn ry = ip->ry[i] - jp->ry;
			real_tn rz = ip->rz[i] - jp->rz;
			real_tn vx = ip->vx[i] - jp->vx;
			real_tn vy = ip->vy[i] - jp->vy;
			real_tn vz = ip->vz[i] - jp->vz;

			real_tn rr = ee;
			rr += rx * rx;
			rr += ry * ry;
			rr += rz * rz;

			real_tn rv = rx * vx;
			rv += ry * vy;
			rv += rz * vz;

			real_tn vv = vx * vx;
			vv += vy * vy;
			vv += vz * vz;

			real_tn inv_r2 = rsqrt(rr);
			m_r3 *= inv_r2;
			inv_r2 *= inv_r2;
			m_r3 *= 2 * inv_r2;

			real_tn m_r5 = m_r3 * inv_r2;
			m_r3 += vv * inv_r2;
			rv *= eta * rsqrt(m_r3);
			m_r5 += m_r3 * inv_r2;
			m_r3 -= m_r5 * rv;

			m_r3 = (rr > ee) ? (m_r3):(0);
			m_r3 = (ip->m[i] == 0 || jp->m == 0) ? (0):(m_r3);

//			jp->w2_a = fmax(m_r3, jp->w2_a);
//			jp->w2_b += m_r3;

			ip->w2_a[i] = fmax(m_r3, ip->w2_a[i]);
			ip->w2_b[i] += m_r3;

			simd_shuff_Tstep_Data(jp);
		}
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
tstep_kernel_rectangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	const real_t eta,
	global real_t __iw2_a[],
	global real_t __iw2_b[],
	global real_t __jw2_a[],
	global real_t __jw2_b[])
{
	local Tstep_Data_SoA _ip[NWARPS];
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t lane = lid % NLANES;
	uint_t warp = lid / NLANES;
	for (uint_t ii = SIMD * WGSIZE * grp;
				ii < ni;
				ii += SIMD * WGSIZE * ngrps) {
		Tstep_Data ip = {{0}};
		read_Tstep_Data(
			ii, lid, &ip,
			ni, __im, __ie2, __irdot
		);
		barrier(CLK_LOCAL_MEM_FENCE);
		_ip[warp].m[lane] = ip.m;
		_ip[warp].e2[lane] = ip.e2;
		_ip[warp].rx[lane] = ip.rx;
		_ip[warp].ry[lane] = ip.ry;
		_ip[warp].rz[lane] = ip.rz;
		_ip[warp].vx[lane] = ip.vx;
		_ip[warp].vy[lane] = ip.vy;
		_ip[warp].vz[lane] = ip.vz;
		_ip[warp].w2_a[lane] = ip.w2_a;
		_ip[warp].w2_b[lane] = ip.w2_b;
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint_t jj = 0;
					jj < nj;
					jj += SIMD * WGSIZE) {
			Tstep_Data jp = {{0}};
			read_Tstep_Data(
				jj, lid, &jp,
				nj, __jm, __je2, __jrdot
			);

			for (uint_t w = 0; w < NWARPS; ++w) {
				p2p_tstep_kernel_core(eta, lane, &jp, &_ip[(warp+w)%NWARPS]);
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			for (uint_t k = 0, kk = jj + lid;
						k < SIMD;
						k += 1, kk += WGSIZE) {
				if (kk < nj) {
					atomic_fmax(&__jw2_a[kk], jp._w2_a[k]);
					atomic_fadd(&__jw2_b[kk], jp._w2_b[k]);
				}
			}
		}

		ip.w2_a = _ip[warp].w2_a[lane];
		ip.w2_b = _ip[warp].w2_b[lane];
		for (uint_t k = 0, kk = ii + lid;
					k < SIMD;
					k += 1, kk += WGSIZE) {
			if (kk < ni) {
				__iw2_a[kk]  = fmax(ip._w2_a[k], __iw2_a[kk]);
				__iw2_b[kk] += ip._w2_b[k];
			}
		}
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
tstep_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const real_t eta,
	global real_t __iw2_a[],
	global real_t __iw2_b[])
{
	// ------------------------------------------------------------------------
	const uint_t nj = ni;
	global const real_t *__jm = __im;
	global const real_t *__je2 = __ie2;
	global const real_t *__jrdot = __irdot;
	global real_t *__jw2_a = __iw2_a;
	global real_t *__jw2_b = __iw2_b;
	// ------------------------------------------------------------------------

	local Tstep_Data_SoA _ip[NWARPS];
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t lane = lid % NLANES;
	uint_t warp = lid / NLANES;
	for (uint_t ii = SIMD * WGSIZE * grp;
				ii < ni;
				ii += SIMD * WGSIZE * ngrps) {
		Tstep_Data ip = {{0}};
		read_Tstep_Data(
			ii, lid, &ip,
			ni, __im, __ie2, __irdot
		);
		barrier(CLK_LOCAL_MEM_FENCE);
		_ip[warp].m[lane] = ip.m;
		_ip[warp].e2[lane] = ip.e2;
		_ip[warp].rx[lane] = ip.rx;
		_ip[warp].ry[lane] = ip.ry;
		_ip[warp].rz[lane] = ip.rz;
		_ip[warp].vx[lane] = ip.vx;
		_ip[warp].vy[lane] = ip.vy;
		_ip[warp].vz[lane] = ip.vz;
		_ip[warp].w2_a[lane] = ip.w2_a;
		_ip[warp].w2_b[lane] = ip.w2_b;
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint_t jj = 0;
					jj < nj;
					jj += SIMD * WGSIZE) {
			Tstep_Data jp = {{0}};
			read_Tstep_Data(
				jj, lid, &jp,
				nj, __jm, __je2, __jrdot
			);

			for (uint_t w = 0; w < NWARPS; ++w) {
				tstep_kernel_core(eta, lane, &jp, &_ip[(warp+w)%NWARPS]);
				barrier(CLK_LOCAL_MEM_FENCE);
			}

//			for (uint_t k = 0, kk = jj + lid;
//						k < SIMD;
//						k += 1, kk += WGSIZE) {
//				if (kk < nj) {
//					atomic_fmax(&__jw2_a[kk], jp._w2_a[k]);
//					atomic_fadd(&__jw2_b[kk], jp._w2_b[k]);
//				}
//			}
		}

		ip.w2_a = _ip[warp].w2_a[lane];
		ip.w2_b = _ip[warp].w2_b[lane];
		for (uint_t k = 0, kk = ii + lid;
					k < SIMD;
					k += 1, kk += WGSIZE) {
			if (kk < ni) {
				__iw2_a[kk]  = fmax(ip._w2_a[k], __iw2_a[kk]);
				__iw2_b[kk] += ip._w2_b[k];
			}
		}
	}
}

