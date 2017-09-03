#include "tstep_kernel_common.h"


static inline void
p2p_tstep_kernel_core(
	const real_t eta,
	uint_t lid,
	concat(Tstep_Data, WPT) *jp,
	local concat(Tstep_Data, WGSIZE) *ip)
// flop count: 43
{
	#pragma unroll
	for (uint_t w = 0; w < WGSIZE; w += NLANES) {
		for (uint_t l = w; l < w + NLANES; ++l) {
			uint_t i = lid^l;
			for (uint_t j = 0; j < WPT; ++j) {
				for (uint_t k = 0; k < SIMD; ++k) {
					real_tn m_r3 = ip->m[i] + jp->m[j];
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

					m_r3 = (ip->m[i] == 0 || jp->m[j] == 0) ? (0):(m_r3);

					jp->w2_a[j] = fmax(m_r3, jp->w2_a[j]);
					jp->w2_b[j] += m_r3;

					ip->w2_a[i] = fmax(m_r3, ip->w2_a[i]);
					ip->w2_b[i] += m_r3;

					simd_shuff_Tstep_Data(j, jp);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


static inline void
tstep_kernel_core(
	const real_t eta,
	uint_t lid,
	concat(Tstep_Data, WPT) *jp,
	local concat(Tstep_Data, WGSIZE) *ip)
// flop count: 42
{
	#pragma unroll
	for (uint_t w = 0; w < WGSIZE; w += NLANES) {
		for (uint_t l = w; l < w + NLANES; ++l) {
			uint_t i = lid^l;
			for (uint_t j = 0; j < WPT; ++j) {
				for (uint_t k = 0; k < SIMD; ++k) {
					real_tn m_r3 = ip->m[i] + jp->m[j];
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
					m_r3 = (ip->m[i] == 0 || jp->m[j] == 0) ? (0):(m_r3);

					jp->w2_a[j] = fmax(m_r3, jp->w2_a[j]);
					jp->w2_b[j] += m_r3;

//					ip->w2_a[i] = fmax(m_r3, ip->w2_a[i]);
//					ip->w2_b[i] += m_r3;

					simd_shuff_Tstep_Data(j, jp);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
tstep_kernel_rectangle(
	const real_t eta,
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	global real_t __iw2_a[],
	global real_t __iw2_b[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	global real_t __jw2_a[],
	global real_t __jw2_b[])
{
	local concat(Tstep_Data, WGSIZE) _ip;
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t block = WGSIZE * SIMD * WPT;
	for (uint_t jj = block * grp;
				jj < nj;
				jj += block * ngrps) {
		concat(Tstep_Data, WPT) jp = {{{0}}};
		concat(load_Tstep_Data, WPT)(
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
				concat(Tstep_Data, 1) ip = {{{0}}};
				concat(load_Tstep_Data, 1)(
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
				_ip.w2_a[lid] = ip.w2_a[0];
				_ip.w2_b[lid] = ip.w2_b[0];

				if (ii != jj) {
					p2p_tstep_kernel_core(eta, lid, &jp, &_ip);
				} else {
					p2p_tstep_kernel_core(eta, lid, &jp, &_ip);
				}

				ip.w2_a[0] = _ip.w2_a[lid];
				ip.w2_b[0] = _ip.w2_b[lid];
				concat(store_Tstep_Data, 1)(
					&ip, ii + ilid, WGSIZE, SIMD,
					ni, __iw2_a, __iw2_b
				);
			}
		}

		concat(store_Tstep_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jw2_a, __jw2_b
		);
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
tstep_kernel_triangle(
	const real_t eta,
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
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

	local concat(Tstep_Data, WGSIZE) _ip;
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t block = WGSIZE * SIMD * WPT;
	for (uint_t jj = block * grp;
				jj < nj;
				jj += block * ngrps) {
		concat(Tstep_Data, WPT) jp = {{{0}}};
		concat(load_Tstep_Data, WPT)(
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
				concat(Tstep_Data, 1) ip = {{{0}}};
				concat(load_Tstep_Data, 1)(
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
				_ip.w2_a[lid] = ip.w2_a[0];
				_ip.w2_b[lid] = ip.w2_b[0];

				if (ii != jj) {
					p2p_tstep_kernel_core(eta, lid, &jp, &_ip);
				} else {
					tstep_kernel_core(eta, lid, &jp, &_ip);
				}

				ip.w2_a[0] = _ip.w2_a[lid];
				ip.w2_b[0] = _ip.w2_b[lid];
				concat(store_Tstep_Data, 1)(
					&ip, ii + ilid, WGSIZE, SIMD,
					ni, __iw2_a, __iw2_b
				);
			}
		}

		concat(store_Tstep_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jw2_a, __jw2_b
		);
	}
}

