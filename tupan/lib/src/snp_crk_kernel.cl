#include "snp_crk_kernel_common.h"


static inline void
snp_crk_kernel_core(
	uint_t lid,
	local Snp_Crk_Data *ip,
	local Snp_Crk_Data *jp)
// flop count: 128
{
	uint_t jwarp = lid / WARP;
	uint_t jlane = lid % WARP;
	uint_t jlid = WARP * jwarp + jlane;

	for (uint_t w = 0; w < WGSIZE/WARP; ++w) {
		uint_t iwarp = jwarp^w;
		for (uint_t l = 0; l < WARP; ++l) {
			uint_t ilane = jlane^l;
			uint_t ilid = WARP * iwarp + ilane;
			for (uint_t ii = 0; ii < LMSIZE; ii += WGSIZE) {
				uint_t i = ii + ilid;
//				real_tn iAx = ip->Ax[i];
//				real_tn iAy = ip->Ay[i];
//				real_tn iAz = ip->Az[i];
//				real_tn iJx = ip->Jx[i];
//				real_tn iJy = ip->Jy[i];
//				real_tn iJz = ip->Jz[i];
//				real_tn iSx = ip->Sx[i];
//				real_tn iSy = ip->Sy[i];
//				real_tn iSz = ip->Sz[i];
//				real_tn iCx = ip->Cx[i];
//				real_tn iCy = ip->Cy[i];
//				real_tn iCz = ip->Cz[i];
				for (uint_t jj = 0; jj < LMSIZE; jj += WGSIZE) {
					uint_t j = jj + jlid;
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
					rr += rx * rx + ry * ry + rz * rz;

					real_tn inv_r3 = rsqrt(rr);
					inv_r3 = (rr > ee) ? (inv_r3):(0);
					real_tn inv_r2 = inv_r3 * inv_r3;
					inv_r3 *= inv_r2;
					inv_r2 *= -3;

					real_tn s1 = rx * vx + ry * vy + rz * vz;
					real_tn s2 = vx * vx + vy * vy + vz * vz;
					real_tn s3 = vx * ax + vy * ay + vz * az;
					s3 *= 3;
					s2 += rx * ax + ry * ay + rz * az;
					s3 += rx * jx + ry * jy + rz * jz;

					#define cq21 ((real_t)(5.0/3.0))
					#define cq31 ((real_t)(8.0/3.0))
					#define cq32 ((real_t)(7.0/3.0))

					const real_tn q1 = inv_r2 * (s1);
					const real_tn q2 = inv_r2 * (s2 + (cq21 * s1) * q1);
					const real_tn q3 = inv_r2 * (s3 + (cq31 * s2) * q1 + (cq32 * s1) * q2);

					const real_tn b3 = 3 * q1;
					const real_tn c3 = 3 * q2;
					const real_tn c2 = 2 * q1;

					jx += b3 * ax + c3 * vx + q3 * rx;
					jy += b3 * ay + c3 * vy + q3 * ry;
					jz += b3 * az + c3 * vz + q3 * rz;

					ax += c2 * vx + q2 * rx;
					ay += c2 * vy + q2 * ry;
					az += c2 * vz + q2 * rz;

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
//					iAx -= jm_r3 * rx;
//					iAy -= jm_r3 * ry;
//					iAz -= jm_r3 * rz;
//					iJx -= jm_r3 * vx;
//					iJy -= jm_r3 * vy;
//					iJz -= jm_r3 * vz;
//					iSx -= jm_r3 * ax;
//					iSy -= jm_r3 * ay;
//					iSz -= jm_r3 * az;
//					iCx -= jm_r3 * jx;
//					iCy -= jm_r3 * jy;
//					iCz -= jm_r3 * jz;
				}
//				ip->Ax[i] = iAx;
//				ip->Ay[i] = iAy;
//				ip->Az[i] = iAz;
//				ip->Jx[i] = iJx;
//				ip->Jy[i] = iJy;
//				ip->Jz[i] = iJz;
//				ip->Sx[i] = iSx;
//				ip->Sy[i] = iSy;
//				ip->Sz[i] = iSz;
//				ip->Cx[i] = iCx;
//				ip->Cy[i] = iCy;
//				ip->Cz[i] = iCz;
			}
		}
	}
}


static inline void
snp_crk_kernel_impl(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	global real_t __iadot[],
	global real_t __jadot[],
	local Snp_Crk_Data *ip,
	local Snp_Crk_Data *jp)
{
	uint_t lid = get_local_id(0);
	for (uint_t ii = LMSIZE * SIMD * get_group_id(0);
				ii < ni;
				ii += LMSIZE * SIMD * get_num_groups(0)) {
		uint_t iN = min((uint_t)(LMSIZE * SIMD), (ni - ii));
		for (uint_t kk = 0; kk < LMSIZE; kk += WGSIZE) {
			uint_t k = kk + lid;
			ip->m[k] = (real_tn)(0);
			ip->e2[k] = (real_tn)(0);
			ip->rx[k] = (real_tn)(0);
			ip->ry[k] = (real_tn)(0);
			ip->rz[k] = (real_tn)(0);
			ip->vx[k] = (real_tn)(0);
			ip->vy[k] = (real_tn)(0);
			ip->vz[k] = (real_tn)(0);
			ip->ax[k] = (real_tn)(0);
			ip->ay[k] = (real_tn)(0);
			ip->az[k] = (real_tn)(0);
			ip->jx[k] = (real_tn)(0);
			ip->jy[k] = (real_tn)(0);
			ip->jz[k] = (real_tn)(0);
			ip->Ax[k] = (real_tn)(0);
			ip->Ay[k] = (real_tn)(0);
			ip->Az[k] = (real_tn)(0);
			ip->Jx[k] = (real_tn)(0);
			ip->Jy[k] = (real_tn)(0);
			ip->Jz[k] = (real_tn)(0);
			ip->Sx[k] = (real_tn)(0);
			ip->Sy[k] = (real_tn)(0);
			ip->Sz[k] = (real_tn)(0);
			ip->Cx[k] = (real_tn)(0);
			ip->Cy[k] = (real_tn)(0);
			ip->Cz[k] = (real_tn)(0);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		async_work_group_copy(ip->_m, __im+ii, iN, 0);
		async_work_group_copy(ip->_e2, __ie2+ii, iN, 0);
		async_work_group_copy(ip->_rx, __irdot+(0*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_ry, __irdot+(0*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_rz, __irdot+(0*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_vx, __irdot+(1*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_vy, __irdot+(1*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_vz, __irdot+(1*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_ax, __irdot+(2*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_ay, __irdot+(2*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_az, __irdot+(2*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_jx, __irdot+(3*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_jy, __irdot+(3*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_jz, __irdot+(3*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_Ax, __iadot+(0*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_Ay, __iadot+(0*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_Az, __iadot+(0*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_Jx, __iadot+(1*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_Jy, __iadot+(1*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_Jz, __iadot+(1*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_Sx, __iadot+(2*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_Sy, __iadot+(2*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_Sz, __iadot+(2*NDIM+2)*ni+ii, iN, 0);
		async_work_group_copy(ip->_Cx, __iadot+(3*NDIM+0)*ni+ii, iN, 0);
		async_work_group_copy(ip->_Cy, __iadot+(3*NDIM+1)*ni+ii, iN, 0);
		async_work_group_copy(ip->_Cz, __iadot+(3*NDIM+2)*ni+ii, iN, 0);
		for (uint_t jj = 0;
					jj < nj;
					jj += LMSIZE * SIMD) {
			uint_t jN = min((uint_t)(LMSIZE * SIMD), (nj - jj));
			for (uint_t kk = 0; kk < LMSIZE; kk += WGSIZE) {
				uint_t k = kk + lid;
				jp->m[k] = (real_tn)(0);
				jp->e2[k] = (real_tn)(0);
				jp->rx[k] = (real_tn)(0);
				jp->ry[k] = (real_tn)(0);
				jp->rz[k] = (real_tn)(0);
				jp->vx[k] = (real_tn)(0);
				jp->vy[k] = (real_tn)(0);
				jp->vz[k] = (real_tn)(0);
				jp->ax[k] = (real_tn)(0);
				jp->ay[k] = (real_tn)(0);
				jp->az[k] = (real_tn)(0);
				jp->jx[k] = (real_tn)(0);
				jp->jy[k] = (real_tn)(0);
				jp->jz[k] = (real_tn)(0);
				jp->Ax[k] = (real_tn)(0);
				jp->Ay[k] = (real_tn)(0);
				jp->Az[k] = (real_tn)(0);
				jp->Jx[k] = (real_tn)(0);
				jp->Jy[k] = (real_tn)(0);
				jp->Jz[k] = (real_tn)(0);
				jp->Sx[k] = (real_tn)(0);
				jp->Sy[k] = (real_tn)(0);
				jp->Sz[k] = (real_tn)(0);
				jp->Cx[k] = (real_tn)(0);
				jp->Cy[k] = (real_tn)(0);
				jp->Cz[k] = (real_tn)(0);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			async_work_group_copy(jp->_m, __jm+jj, jN, 0);
			async_work_group_copy(jp->_e2, __je2+jj, jN, 0);
			async_work_group_copy(jp->_rx, __jrdot+(0*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_ry, __jrdot+(0*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_rz, __jrdot+(0*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_vx, __jrdot+(1*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_vy, __jrdot+(1*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_vz, __jrdot+(1*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_ax, __jrdot+(2*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_ay, __jrdot+(2*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_az, __jrdot+(2*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_jx, __jrdot+(3*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_jy, __jrdot+(3*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_jz, __jrdot+(3*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_Ax, __jadot+(0*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_Ay, __jadot+(0*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_Az, __jadot+(0*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_Jx, __jadot+(1*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_Jy, __jadot+(1*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_Jz, __jadot+(1*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_Sx, __jadot+(2*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_Sy, __jadot+(2*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_Sz, __jadot+(2*NDIM+2)*nj+jj, jN, 0);
			async_work_group_copy(jp->_Cx, __jadot+(3*NDIM+0)*nj+jj, jN, 0);
			async_work_group_copy(jp->_Cy, __jadot+(3*NDIM+1)*nj+jj, jN, 0);
			async_work_group_copy(jp->_Cz, __jadot+(3*NDIM+2)*nj+jj, jN, 0);
			barrier(CLK_LOCAL_MEM_FENCE);
			for (uint_t k = 0; k < SIMD; ++k) {
				snp_crk_kernel_core(lid, jp, ip);
				for (uint_t kk = 0; kk < LMSIZE; kk += WGSIZE) {
					uint_t i = kk + lid;
					shuff(ip->m[i], SIMD);
					shuff(ip->e2[i], SIMD);
					shuff(ip->rx[i], SIMD);
					shuff(ip->ry[i], SIMD);
					shuff(ip->rz[i], SIMD);
					shuff(ip->vx[i], SIMD);
					shuff(ip->vy[i], SIMD);
					shuff(ip->vz[i], SIMD);
					shuff(ip->ax[i], SIMD);
					shuff(ip->ay[i], SIMD);
					shuff(ip->az[i], SIMD);
					shuff(ip->jx[i], SIMD);
					shuff(ip->jy[i], SIMD);
					shuff(ip->jz[i], SIMD);
					shuff(ip->Ax[i], SIMD);
					shuff(ip->Ay[i], SIMD);
					shuff(ip->Az[i], SIMD);
					shuff(ip->Jx[i], SIMD);
					shuff(ip->Jy[i], SIMD);
					shuff(ip->Jz[i], SIMD);
					shuff(ip->Sx[i], SIMD);
					shuff(ip->Sy[i], SIMD);
					shuff(ip->Sz[i], SIMD);
					shuff(ip->Cx[i], SIMD);
					shuff(ip->Cy[i], SIMD);
					shuff(ip->Cz[i], SIMD);
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		async_work_group_copy(__iadot+(0*NDIM+0)*ni+ii, ip->_Ax, iN, 0);
		async_work_group_copy(__iadot+(0*NDIM+1)*ni+ii, ip->_Ay, iN, 0);
		async_work_group_copy(__iadot+(0*NDIM+2)*ni+ii, ip->_Az, iN, 0);
		async_work_group_copy(__iadot+(1*NDIM+0)*ni+ii, ip->_Jx, iN, 0);
		async_work_group_copy(__iadot+(1*NDIM+1)*ni+ii, ip->_Jy, iN, 0);
		async_work_group_copy(__iadot+(1*NDIM+2)*ni+ii, ip->_Jz, iN, 0);
		async_work_group_copy(__iadot+(2*NDIM+0)*ni+ii, ip->_Sx, iN, 0);
		async_work_group_copy(__iadot+(2*NDIM+1)*ni+ii, ip->_Sy, iN, 0);
		async_work_group_copy(__iadot+(2*NDIM+2)*ni+ii, ip->_Sz, iN, 0);
		async_work_group_copy(__iadot+(3*NDIM+0)*ni+ii, ip->_Cx, iN, 0);
		async_work_group_copy(__iadot+(3*NDIM+1)*ni+ii, ip->_Cy, iN, 0);
		async_work_group_copy(__iadot+(3*NDIM+2)*ni+ii, ip->_Cz, iN, 0);
	}
}


kernel __attribute__((reqd_work_group_size(WGSIZE, 1, 1))) void
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
	local Snp_Crk_Data _ip;
	local Snp_Crk_Data _jp;

	snp_crk_kernel_impl(
		ni, __im, __ie2, __irdot,
		nj, __jm, __je2, __jrdot,
		__iadot, __jadot,
		&_ip, &_jp
	);

	snp_crk_kernel_impl(
		nj, __jm, __je2, __jrdot,
		ni, __im, __ie2, __irdot,
		__jadot, __iadot,
		&_jp, &_ip
	);
}


kernel __attribute__((reqd_work_group_size(WGSIZE, 1, 1))) void
snp_crk_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	global real_t __iadot[])
{
	local Snp_Crk_Data _ip;
	local Snp_Crk_Data _jp;

	snp_crk_kernel_impl(
		ni, __im, __ie2, __irdot,
		ni, __im, __ie2, __irdot,
		__iadot, __iadot,
		&_ip, &_jp
	);
}

