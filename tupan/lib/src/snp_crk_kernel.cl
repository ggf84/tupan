#include "snp_crk_kernel_common.h"


static inline void
p2p_snp_crk_kernel_core(
	uint_t lid,
	concat(Snp_Crk_Data, WPT) *jp,
	local concat(Snp_Crk_Data, WGSIZE) *ip)
// flop count: 153
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
					real_tn ax = ip->ax[i] - jp->ax[j];
					real_tn ay = ip->ay[i] - jp->ay[j];
					real_tn az = ip->az[i] - jp->az[j];
					real_tn jx = ip->jx[i] - jp->jx[j];
					real_tn jy = ip->jy[i] - jp->jy[j];
					real_tn jz = ip->jz[i] - jp->jz[j];

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
					real_tn s2  = vx * vx;
							s2 += vy * vy;
							s2 += vz * vz;
					real_tn s3  = vx * ax;
							s3 += vy * ay;
							s3 += vz * az;
					s3 *= 3;
					s2 += rx * ax;
					s2 += ry * ay;
					s2 += rz * az;
					s3 += rx * jx;
					s3 += ry * jy;
					s3 += rz * jz;

					#define cq21 ((real_t)(5.0/3.0))
					#define cq31 ((real_t)(8.0/3.0))
					#define cq32 ((real_t)(7.0/3.0))

					const real_tn q1 = inv_r2 * (s1);
					const real_tn q2 = inv_r2 * (s2 + (cq21 * s1) * q1);
					const real_tn q3 = inv_r2 * (s3 + (cq31 * s2) * q1 + (cq32 * s1) * q2);

					const real_tn b3 = 3 * q1;
					const real_tn c3 = 3 * q2;
					const real_tn c2 = 2 * q1;

					jx += b3 * ax;
					jy += b3 * ay;
					jz += b3 * az;
					jx += c3 * vx;
					jy += c3 * vy;
					jz += c3 * vz;
					jx += q3 * rx;
					jy += q3 * ry;
					jz += q3 * rz;

					ax += c2 * vx;
					ay += c2 * vy;
					az += c2 * vz;
					ax += q2 * rx;
					ay += q2 * ry;
					az += q2 * rz;

					vx += q1 * rx;
					vy += q1 * ry;
					vz += q1 * rz;

					real_tn im_r3 = ip->m[i] * inv_r3;
					jp->Ax[j] += im_r3 * rx;
					jp->Ay[j] += im_r3 * ry;
					jp->Az[j] += im_r3 * rz;
					jp->Jx[j] += im_r3 * vx;
					jp->Jy[j] += im_r3 * vy;
					jp->Jz[j] += im_r3 * vz;
					jp->Sx[j] += im_r3 * ax;
					jp->Sy[j] += im_r3 * ay;
					jp->Sz[j] += im_r3 * az;
					jp->Cx[j] += im_r3 * jx;
					jp->Cy[j] += im_r3 * jy;
					jp->Cz[j] += im_r3 * jz;

					real_tn jm_r3 = jp->m[j] * inv_r3;
					ip->Ax[i] += jm_r3 * rx;
					ip->Ay[i] += jm_r3 * ry;
					ip->Az[i] += jm_r3 * rz;
					ip->Jx[i] += jm_r3 * vx;
					ip->Jy[i] += jm_r3 * vy;
					ip->Jz[i] += jm_r3 * vz;
					ip->Sx[i] += jm_r3 * ax;
					ip->Sy[i] += jm_r3 * ay;
					ip->Sz[i] += jm_r3 * az;
					ip->Cx[i] += jm_r3 * jx;
					ip->Cy[i] += jm_r3 * jy;
					ip->Cz[i] += jm_r3 * jz;

					simd_shuff_Snp_Crk_Data(j, jp);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


static inline void
snp_crk_kernel_core(
	uint_t lid,
	concat(Snp_Crk_Data, WPT) *jp,
	local concat(Snp_Crk_Data, WGSIZE) *ip)
// flop count: 128
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
					real_tn ax = ip->ax[i] - jp->ax[j];
					real_tn ay = ip->ay[i] - jp->ay[j];
					real_tn az = ip->az[i] - jp->az[j];
					real_tn jx = ip->jx[i] - jp->jx[j];
					real_tn jy = ip->jy[i] - jp->jy[j];
					real_tn jz = ip->jz[i] - jp->jz[j];

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
					real_tn s2  = vx * vx;
							s2 += vy * vy;
							s2 += vz * vz;
					real_tn s3  = vx * ax;
							s3 += vy * ay;
							s3 += vz * az;
					s3 *= 3;
					s2 += rx * ax;
					s2 += ry * ay;
					s2 += rz * az;
					s3 += rx * jx;
					s3 += ry * jy;
					s3 += rz * jz;

					#define cq21 ((real_t)(5.0/3.0))
					#define cq31 ((real_t)(8.0/3.0))
					#define cq32 ((real_t)(7.0/3.0))

					const real_tn q1 = inv_r2 * (s1);
					const real_tn q2 = inv_r2 * (s2 + (cq21 * s1) * q1);
					const real_tn q3 = inv_r2 * (s3 + (cq31 * s2) * q1 + (cq32 * s1) * q2);

					const real_tn b3 = 3 * q1;
					const real_tn c3 = 3 * q2;
					const real_tn c2 = 2 * q1;

					jx += b3 * ax;
					jy += b3 * ay;
					jz += b3 * az;
					jx += c3 * vx;
					jy += c3 * vy;
					jz += c3 * vz;
					jx += q3 * rx;
					jy += q3 * ry;
					jz += q3 * rz;

					ax += c2 * vx;
					ay += c2 * vy;
					az += c2 * vz;
					ax += q2 * rx;
					ay += q2 * ry;
					az += q2 * rz;

					vx += q1 * rx;
					vy += q1 * ry;
					vz += q1 * rz;

					real_tn im_r3 = ip->m[i] * inv_r3;
					jp->Ax[j] += im_r3 * rx;
					jp->Ay[j] += im_r3 * ry;
					jp->Az[j] += im_r3 * rz;
					jp->Jx[j] += im_r3 * vx;
					jp->Jy[j] += im_r3 * vy;
					jp->Jz[j] += im_r3 * vz;
					jp->Sx[j] += im_r3 * ax;
					jp->Sy[j] += im_r3 * ay;
					jp->Sz[j] += im_r3 * az;
					jp->Cx[j] += im_r3 * jx;
					jp->Cy[j] += im_r3 * jy;
					jp->Cz[j] += im_r3 * jz;

//					real_tn jm_r3 = jp->m[j] * inv_r3;
//					ip->Ax[i] += jm_r3 * rx;
//					ip->Ay[i] += jm_r3 * ry;
//					ip->Az[i] += jm_r3 * rz;
//					ip->Jx[i] += jm_r3 * vx;
//					ip->Jy[i] += jm_r3 * vy;
//					ip->Jz[i] += jm_r3 * vz;
//					ip->Sx[i] += jm_r3 * ax;
//					ip->Sy[i] += jm_r3 * ay;
//					ip->Sz[i] += jm_r3 * az;
//					ip->Cx[i] += jm_r3 * jx;
//					ip->Cy[i] += jm_r3 * jy;
//					ip->Cz[i] += jm_r3 * jz;

					simd_shuff_Snp_Crk_Data(j, jp);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
snp_crk_kernel_rectangle(
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
	local concat(Snp_Crk_Data, WGSIZE) _ip;
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t block = WGSIZE * SIMD * WPT;
	for (uint_t jj = block * grp;
				jj < nj;
				jj += block * ngrps) {
		concat(Snp_Crk_Data, WPT) jp = {{{0}}};
		concat(load_Snp_Crk_Data, WPT)(
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
				concat(Snp_Crk_Data, 1) ip = {{{0}}};
				concat(load_Snp_Crk_Data, 1)(
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
				_ip.Ax[lid] = ip.Ax[0];
				_ip.Ay[lid] = ip.Ay[0];
				_ip.Az[lid] = ip.Az[0];
				_ip.Jx[lid] = ip.Jx[0];
				_ip.Jy[lid] = ip.Jy[0];
				_ip.Jz[lid] = ip.Jz[0];
				_ip.Sx[lid] = ip.Sx[0];
				_ip.Sy[lid] = ip.Sy[0];
				_ip.Sz[lid] = ip.Sz[0];
				_ip.Cx[lid] = ip.Cx[0];
				_ip.Cy[lid] = ip.Cy[0];
				_ip.Cz[lid] = ip.Cz[0];

				if (ii != jj) {
					p2p_snp_crk_kernel_core(lid, &jp, &_ip);
				} else {
					p2p_snp_crk_kernel_core(lid, &jp, &_ip);
				}

				ip.Ax[0] = -_ip.Ax[lid];
				ip.Ay[0] = -_ip.Ay[lid];
				ip.Az[0] = -_ip.Az[lid];
				ip.Jx[0] = -_ip.Jx[lid];
				ip.Jy[0] = -_ip.Jy[lid];
				ip.Jz[0] = -_ip.Jz[lid];
				ip.Sx[0] = -_ip.Sx[lid];
				ip.Sy[0] = -_ip.Sy[lid];
				ip.Sz[0] = -_ip.Sz[lid];
				ip.Cx[0] = -_ip.Cx[lid];
				ip.Cy[0] = -_ip.Cy[lid];
				ip.Cz[0] = -_ip.Cz[lid];
				concat(store_Snp_Crk_Data, 1)(
					&ip, ii + ilid, WGSIZE, SIMD,
					ni, __iadot
				);
			}
		}

		concat(store_Snp_Crk_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jadot
		);
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
snp_crk_kernel_triangle(
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

	local concat(Snp_Crk_Data, WGSIZE) _ip;
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t block = WGSIZE * SIMD * WPT;
	for (uint_t jj = block * grp;
				jj < nj;
				jj += block * ngrps) {
		concat(Snp_Crk_Data, WPT) jp = {{{0}}};
		concat(load_Snp_Crk_Data, WPT)(
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
				concat(Snp_Crk_Data, 1) ip = {{{0}}};
				concat(load_Snp_Crk_Data, 1)(
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
				_ip.Ax[lid] = ip.Ax[0];
				_ip.Ay[lid] = ip.Ay[0];
				_ip.Az[lid] = ip.Az[0];
				_ip.Jx[lid] = ip.Jx[0];
				_ip.Jy[lid] = ip.Jy[0];
				_ip.Jz[lid] = ip.Jz[0];
				_ip.Sx[lid] = ip.Sx[0];
				_ip.Sy[lid] = ip.Sy[0];
				_ip.Sz[lid] = ip.Sz[0];
				_ip.Cx[lid] = ip.Cx[0];
				_ip.Cy[lid] = ip.Cy[0];
				_ip.Cz[lid] = ip.Cz[0];

				if (ii != jj) {
					p2p_snp_crk_kernel_core(lid, &jp, &_ip);
				} else {
					snp_crk_kernel_core(lid, &jp, &_ip);
				}

				ip.Ax[0] = -_ip.Ax[lid];
				ip.Ay[0] = -_ip.Ay[lid];
				ip.Az[0] = -_ip.Az[lid];
				ip.Jx[0] = -_ip.Jx[lid];
				ip.Jy[0] = -_ip.Jy[lid];
				ip.Jz[0] = -_ip.Jz[lid];
				ip.Sx[0] = -_ip.Sx[lid];
				ip.Sy[0] = -_ip.Sy[lid];
				ip.Sz[0] = -_ip.Sz[lid];
				ip.Cx[0] = -_ip.Cx[lid];
				ip.Cy[0] = -_ip.Cy[lid];
				ip.Cz[0] = -_ip.Cz[lid];
				concat(store_Snp_Crk_Data, 1)(
					&ip, ii + ilid, WGSIZE, SIMD,
					ni, __iadot
				);
			}
		}

		concat(store_Snp_Crk_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jadot
		);
	}
}

