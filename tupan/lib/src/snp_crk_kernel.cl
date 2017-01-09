#include "snp_crk_kernel_common.h"


void
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
	local Snp_Crk_Data _jp[])
{
	uint_t lid = get_local_id(0);
	uint_t wid = get_group_id(0);
	uint_t wsize = get_num_groups(0);

	for (uint_t iii = SIMD * LSIZE * wid;
				iii < ni;
				iii += SIMD * LSIZE * wsize) {
		Snp_Crk_Data ip = {{0}};
		#pragma unroll SIMD
		for (uint_t i = 0, ii = iii + lid;
					i < SIMD && ii < ni;
					++i, ii += LSIZE) {
			ip._m[i] = __im[ii];
			ip._e2[i] = __ie2[ii];
			ip._rx[i] = __irdot[(0*NDIM+0)*ni + ii];
			ip._ry[i] = __irdot[(0*NDIM+1)*ni + ii];
			ip._rz[i] = __irdot[(0*NDIM+2)*ni + ii];
			ip._vx[i] = __irdot[(1*NDIM+0)*ni + ii];
			ip._vy[i] = __irdot[(1*NDIM+1)*ni + ii];
			ip._vz[i] = __irdot[(1*NDIM+2)*ni + ii];
			ip._ax[i] = __irdot[(2*NDIM+0)*ni + ii];
			ip._ay[i] = __irdot[(2*NDIM+1)*ni + ii];
			ip._az[i] = __irdot[(2*NDIM+2)*ni + ii];
			ip._jx[i] = __irdot[(3*NDIM+0)*ni + ii];
			ip._jy[i] = __irdot[(3*NDIM+1)*ni + ii];
			ip._jz[i] = __irdot[(3*NDIM+2)*ni + ii];
			ip._Ax[i] = __iadot[(0*NDIM+0)*ni + ii];
			ip._Ay[i] = __iadot[(0*NDIM+1)*ni + ii];
			ip._Az[i] = __iadot[(0*NDIM+2)*ni + ii];
			ip._Jx[i] = __iadot[(1*NDIM+0)*ni + ii];
			ip._Jy[i] = __iadot[(1*NDIM+1)*ni + ii];
			ip._Jz[i] = __iadot[(1*NDIM+2)*ni + ii];
			ip._Sx[i] = __iadot[(2*NDIM+0)*ni + ii];
			ip._Sy[i] = __iadot[(2*NDIM+1)*ni + ii];
			ip._Sz[i] = __iadot[(2*NDIM+2)*ni + ii];
			ip._Cx[i] = __iadot[(3*NDIM+0)*ni + ii];
			ip._Cy[i] = __iadot[(3*NDIM+1)*ni + ii];
			ip._Cz[i] = __iadot[(3*NDIM+2)*ni + ii];
		}
		uint_t j0 = 0;
		uint_t j1 = 0;
		#pragma unroll
		for (uint_t jlsize = LSIZE;
					jlsize > 0;
					jlsize >>= 1) {
			j0 = j1 + lid % jlsize;
			j1 = jlsize * (nj/jlsize);
			for (uint_t jj = j0;
						jj < j1;
						jj += jlsize) {
				Snp_Crk_Data jp = {{0}};
				jp.m = (real_tn)(__jm[jj]);
				jp.e2 = (real_tn)(__je2[jj]);
				jp.rx = (real_tn)(__jrdot[(0*NDIM+0)*nj + jj]);
				jp.ry = (real_tn)(__jrdot[(0*NDIM+1)*nj + jj]);
				jp.rz = (real_tn)(__jrdot[(0*NDIM+2)*nj + jj]);
				jp.vx = (real_tn)(__jrdot[(1*NDIM+0)*nj + jj]);
				jp.vy = (real_tn)(__jrdot[(1*NDIM+1)*nj + jj]);
				jp.vz = (real_tn)(__jrdot[(1*NDIM+2)*nj + jj]);
				jp.ax = (real_tn)(__jrdot[(2*NDIM+0)*nj + jj]);
				jp.ay = (real_tn)(__jrdot[(2*NDIM+1)*nj + jj]);
				jp.az = (real_tn)(__jrdot[(2*NDIM+2)*nj + jj]);
				jp.jx = (real_tn)(__jrdot[(3*NDIM+0)*nj + jj]);
				jp.jy = (real_tn)(__jrdot[(3*NDIM+1)*nj + jj]);
				jp.jz = (real_tn)(__jrdot[(3*NDIM+2)*nj + jj]);
				jp.Ax = (real_tn)(__jadot[(0*NDIM+0)*nj + jj]);
				jp.Ay = (real_tn)(__jadot[(0*NDIM+1)*nj + jj]);
				jp.Az = (real_tn)(__jadot[(0*NDIM+2)*nj + jj]);
				jp.Jx = (real_tn)(__jadot[(1*NDIM+0)*nj + jj]);
				jp.Jy = (real_tn)(__jadot[(1*NDIM+1)*nj + jj]);
				jp.Jz = (real_tn)(__jadot[(1*NDIM+2)*nj + jj]);
				jp.Sx = (real_tn)(__jadot[(2*NDIM+0)*nj + jj]);
				jp.Sy = (real_tn)(__jadot[(2*NDIM+1)*nj + jj]);
				jp.Sz = (real_tn)(__jadot[(2*NDIM+2)*nj + jj]);
				jp.Cx = (real_tn)(__jadot[(3*NDIM+0)*nj + jj]);
				jp.Cy = (real_tn)(__jadot[(3*NDIM+1)*nj + jj]);
				jp.Cz = (real_tn)(__jadot[(3*NDIM+2)*nj + jj]);
				barrier(CLK_LOCAL_MEM_FENCE);
				_jp[lid] = jp;
				barrier(CLK_LOCAL_MEM_FENCE);
				#pragma unroll 8
				for (uint_t j = 0; j < jlsize; ++j) {
					ip = snp_crk_kernel_core(ip, _jp[j]);
				}
			}
		}
		#pragma unroll SIMD
		for (uint_t i = 0, ii = iii + lid;
					i < SIMD && ii < ni;
					++i, ii += LSIZE) {
			__iadot[(0*NDIM+0)*ni + ii] = ip._Ax[i];
			__iadot[(0*NDIM+1)*ni + ii] = ip._Ay[i];
			__iadot[(0*NDIM+2)*ni + ii] = ip._Az[i];
			__iadot[(1*NDIM+0)*ni + ii] = ip._Jx[i];
			__iadot[(1*NDIM+1)*ni + ii] = ip._Jy[i];
			__iadot[(1*NDIM+2)*ni + ii] = ip._Jz[i];
			__iadot[(2*NDIM+0)*ni + ii] = ip._Sx[i];
			__iadot[(2*NDIM+1)*ni + ii] = ip._Sy[i];
			__iadot[(2*NDIM+2)*ni + ii] = ip._Sz[i];
			__iadot[(3*NDIM+0)*ni + ii] = ip._Cx[i];
			__iadot[(3*NDIM+1)*ni + ii] = ip._Cy[i];
			__iadot[(3*NDIM+2)*ni + ii] = ip._Cz[i];
		}
	}
}


kernel void
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
	local Snp_Crk_Data _jp[LSIZE];

	snp_crk_kernel_impl(
		ni, __im, __ie2, __irdot,
		nj, __jm, __je2, __jrdot,
		__iadot, __jadot,
		_jp
	);

	snp_crk_kernel_impl(
		nj, __jm, __je2, __jrdot,
		ni, __im, __ie2, __irdot,
		__jadot, __iadot,
		_jp
	);
}


kernel void
snp_crk_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	global real_t __iadot[])
{
	local Snp_Crk_Data _jp[LSIZE];

	snp_crk_kernel_impl(
		ni, __im, __ie2, __irdot,
		ni, __im, __ie2, __irdot,
		__iadot, __iadot,
		_jp
	);
}

