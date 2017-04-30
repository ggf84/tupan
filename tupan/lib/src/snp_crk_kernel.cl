#include "snp_crk_kernel_common.h"


static inline void
p2p_snp_crk_kernel_core(
	uint_t lane,
	Snp_Crk_Data *jp,
	local Snp_Crk_Data_SoA *ip)
// flop count: 153
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
}


static inline void
snp_crk_kernel_core(
	uint_t lane,
	Snp_Crk_Data *jp,
	local Snp_Crk_Data_SoA *ip)
// flop count: 128
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

//				real_tn jm_r3 = jp->m[j] * inv_r3;
//				ip->Ax[i] += jm_r3 * rx;
//				ip->Ay[i] += jm_r3 * ry;
//				ip->Az[i] += jm_r3 * rz;
//				ip->Jx[i] += jm_r3 * vx;
//				ip->Jy[i] += jm_r3 * vy;
//				ip->Jz[i] += jm_r3 * vz;
//				ip->Sx[i] += jm_r3 * ax;
//				ip->Sy[i] += jm_r3 * ay;
//				ip->Sz[i] += jm_r3 * az;
//				ip->Cx[i] += jm_r3 * jx;
//				ip->Cy[i] += jm_r3 * jy;
//				ip->Cz[i] += jm_r3 * jz;

				simd_shuff_Snp_Crk_Data(j, jp);
			}
		}
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
	local Snp_Crk_Data_SoA _ip[NWARPS];
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t lane = lid % NLANES;
	uint_t warp = lid / NLANES;
	for (uint_t jj = 0;
				jj < nj;
				jj += WGSIZE * SIMD * WPT) {
		Snp_Crk_Data jp = {{{0}}};
		read_Snp_Crk_Data(
			&jp, jj + lid, WGSIZE, SIMD * WPT,
			nj, __jm, __je2, __jrdot
		);

		for (uint_t ii = WGSIZE * SIMD * grp;
					ii < ni;
					ii += WGSIZE * SIMD * ngrps) {
			Snp_Crk_Data ip = {{{0}}};
			read_Snp_Crk_Data(
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
			_ip[warp].Ax[lane] = ip.Ax[0];
			_ip[warp].Ay[lane] = ip.Ay[0];
			_ip[warp].Az[lane] = ip.Az[0];
			_ip[warp].Jx[lane] = ip.Jx[0];
			_ip[warp].Jy[lane] = ip.Jy[0];
			_ip[warp].Jz[lane] = ip.Jz[0];
			_ip[warp].Sx[lane] = ip.Sx[0];
			_ip[warp].Sy[lane] = ip.Sy[0];
			_ip[warp].Sz[lane] = ip.Sz[0];
			_ip[warp].Cx[lane] = ip.Cx[0];
			_ip[warp].Cy[lane] = ip.Cy[0];
			_ip[warp].Cz[lane] = ip.Cz[0];

			for (uint_t w = 0; w < NWARPS; ++w) {
				p2p_snp_crk_kernel_core(lane, &jp, &_ip[(warp+w)%NWARPS]);
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			ip.Ax[0] = _ip[warp].Ax[lane];
			ip.Ay[0] = _ip[warp].Ay[lane];
			ip.Az[0] = _ip[warp].Az[lane];
			ip.Jx[0] = _ip[warp].Jx[lane];
			ip.Jy[0] = _ip[warp].Jy[lane];
			ip.Jz[0] = _ip[warp].Jz[lane];
			ip.Sx[0] = _ip[warp].Sx[lane];
			ip.Sy[0] = _ip[warp].Sy[lane];
			ip.Sz[0] = _ip[warp].Sz[lane];
			ip.Cx[0] = _ip[warp].Cx[lane];
			ip.Cy[0] = _ip[warp].Cy[lane];
			ip.Cz[0] = _ip[warp].Cz[lane];
			for (uint_t k = 0, kk = ii + lid;
						k < SIMD;
						k += 1, kk += WGSIZE) {
				if (kk < ni) {
					(__iadot+(0*NDIM+0)*ni)[kk] -= ip._Ax[k];
					(__iadot+(0*NDIM+1)*ni)[kk] -= ip._Ay[k];
					(__iadot+(0*NDIM+2)*ni)[kk] -= ip._Az[k];
					(__iadot+(1*NDIM+0)*ni)[kk] -= ip._Jx[k];
					(__iadot+(1*NDIM+1)*ni)[kk] -= ip._Jy[k];
					(__iadot+(1*NDIM+2)*ni)[kk] -= ip._Jz[k];
					(__iadot+(2*NDIM+0)*ni)[kk] -= ip._Sx[k];
					(__iadot+(2*NDIM+1)*ni)[kk] -= ip._Sy[k];
					(__iadot+(2*NDIM+2)*ni)[kk] -= ip._Sz[k];
					(__iadot+(3*NDIM+0)*ni)[kk] -= ip._Cx[k];
					(__iadot+(3*NDIM+1)*ni)[kk] -= ip._Cy[k];
					(__iadot+(3*NDIM+2)*ni)[kk] -= ip._Cz[k];
				}
			}
		}

		for (uint_t k = 0, kk = jj + lid;
					k < SIMD * WPT;
					k += 1, kk += WGSIZE) {
			if (kk < nj) {
				atomic_fadd(&(__jadot+(0*NDIM+0)*nj)[kk], jp._Ax[k]);
				atomic_fadd(&(__jadot+(0*NDIM+1)*nj)[kk], jp._Ay[k]);
				atomic_fadd(&(__jadot+(0*NDIM+2)*nj)[kk], jp._Az[k]);
				atomic_fadd(&(__jadot+(1*NDIM+0)*nj)[kk], jp._Jx[k]);
				atomic_fadd(&(__jadot+(1*NDIM+1)*nj)[kk], jp._Jy[k]);
				atomic_fadd(&(__jadot+(1*NDIM+2)*nj)[kk], jp._Jz[k]);
				atomic_fadd(&(__jadot+(2*NDIM+0)*nj)[kk], jp._Sx[k]);
				atomic_fadd(&(__jadot+(2*NDIM+1)*nj)[kk], jp._Sy[k]);
				atomic_fadd(&(__jadot+(2*NDIM+2)*nj)[kk], jp._Sz[k]);
				atomic_fadd(&(__jadot+(3*NDIM+0)*nj)[kk], jp._Cx[k]);
				atomic_fadd(&(__jadot+(3*NDIM+1)*nj)[kk], jp._Cy[k]);
				atomic_fadd(&(__jadot+(3*NDIM+2)*nj)[kk], jp._Cz[k]);
			}
		}
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

	local Snp_Crk_Data_SoA _ip[NWARPS];
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t lane = lid % NLANES;
	uint_t warp = lid / NLANES;
	for (uint_t jj = 0;
				jj < nj;
				jj += WGSIZE * SIMD * WPT) {
		Snp_Crk_Data jp = {{{0}}};
		read_Snp_Crk_Data(
			&jp, jj + lid, WGSIZE, SIMD * WPT,
			nj, __jm, __je2, __jrdot
		);

		for (uint_t ii = WGSIZE * SIMD * grp;
					ii < ni;
					ii += WGSIZE * SIMD * ngrps) {
			Snp_Crk_Data ip = {{{0}}};
			read_Snp_Crk_Data(
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
			_ip[warp].Ax[lane] = ip.Ax[0];
			_ip[warp].Ay[lane] = ip.Ay[0];
			_ip[warp].Az[lane] = ip.Az[0];
			_ip[warp].Jx[lane] = ip.Jx[0];
			_ip[warp].Jy[lane] = ip.Jy[0];
			_ip[warp].Jz[lane] = ip.Jz[0];
			_ip[warp].Sx[lane] = ip.Sx[0];
			_ip[warp].Sy[lane] = ip.Sy[0];
			_ip[warp].Sz[lane] = ip.Sz[0];
			_ip[warp].Cx[lane] = ip.Cx[0];
			_ip[warp].Cy[lane] = ip.Cy[0];
			_ip[warp].Cz[lane] = ip.Cz[0];

			for (uint_t w = 0; w < NWARPS; ++w) {
				snp_crk_kernel_core(lane, &jp, &_ip[(warp+w)%NWARPS]);
				barrier(CLK_LOCAL_MEM_FENCE);
			}

//			ip.Ax[0] = _ip[warp].Ax[lane];
//			ip.Ay[0] = _ip[warp].Ay[lane];
//			ip.Az[0] = _ip[warp].Az[lane];
//			ip.Jx[0] = _ip[warp].Jx[lane];
//			ip.Jy[0] = _ip[warp].Jy[lane];
//			ip.Jz[0] = _ip[warp].Jz[lane];
//			ip.Sx[0] = _ip[warp].Sx[lane];
//			ip.Sy[0] = _ip[warp].Sy[lane];
//			ip.Sz[0] = _ip[warp].Sz[lane];
//			ip.Cx[0] = _ip[warp].Cx[lane];
//			ip.Cy[0] = _ip[warp].Cy[lane];
//			ip.Cz[0] = _ip[warp].Cz[lane];
//			for (uint_t k = 0, kk = ii + lid;
//						k < SIMD;
//						k += 1, kk += WGSIZE) {
//				if (kk < ni) {
//					(__iadot+(0*NDIM+0)*ni)[kk] -= ip._Ax[k];
//					(__iadot+(0*NDIM+1)*ni)[kk] -= ip._Ay[k];
//					(__iadot+(0*NDIM+2)*ni)[kk] -= ip._Az[k];
//					(__iadot+(1*NDIM+0)*ni)[kk] -= ip._Jx[k];
//					(__iadot+(1*NDIM+1)*ni)[kk] -= ip._Jy[k];
//					(__iadot+(1*NDIM+2)*ni)[kk] -= ip._Jz[k];
//					(__iadot+(2*NDIM+0)*ni)[kk] -= ip._Sx[k];
//					(__iadot+(2*NDIM+1)*ni)[kk] -= ip._Sy[k];
//					(__iadot+(2*NDIM+2)*ni)[kk] -= ip._Sz[k];
//					(__iadot+(3*NDIM+0)*ni)[kk] -= ip._Cx[k];
//					(__iadot+(3*NDIM+1)*ni)[kk] -= ip._Cy[k];
//					(__iadot+(3*NDIM+2)*ni)[kk] -= ip._Cz[k];
//				}
//			}
		}

		for (uint_t k = 0, kk = jj + lid;
					k < SIMD * WPT;
					k += 1, kk += WGSIZE) {
			if (kk < nj) {
				atomic_fadd(&(__jadot+(0*NDIM+0)*nj)[kk], jp._Ax[k]);
				atomic_fadd(&(__jadot+(0*NDIM+1)*nj)[kk], jp._Ay[k]);
				atomic_fadd(&(__jadot+(0*NDIM+2)*nj)[kk], jp._Az[k]);
				atomic_fadd(&(__jadot+(1*NDIM+0)*nj)[kk], jp._Jx[k]);
				atomic_fadd(&(__jadot+(1*NDIM+1)*nj)[kk], jp._Jy[k]);
				atomic_fadd(&(__jadot+(1*NDIM+2)*nj)[kk], jp._Jz[k]);
				atomic_fadd(&(__jadot+(2*NDIM+0)*nj)[kk], jp._Sx[k]);
				atomic_fadd(&(__jadot+(2*NDIM+1)*nj)[kk], jp._Sy[k]);
				atomic_fadd(&(__jadot+(2*NDIM+2)*nj)[kk], jp._Sz[k]);
				atomic_fadd(&(__jadot+(3*NDIM+0)*nj)[kk], jp._Cx[k]);
				atomic_fadd(&(__jadot+(3*NDIM+1)*nj)[kk], jp._Cy[k]);
				atomic_fadd(&(__jadot+(3*NDIM+2)*nj)[kk], jp._Cz[k]);
			}
		}
	}
}

