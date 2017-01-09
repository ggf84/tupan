#include "phi_kernel_common.h"


void
phi_kernel_impl(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	global real_t __iphi[],
	global real_t __jphi[],
	local Phi_Data _jp[])
{
	uint_t lid = get_local_id(0);
	uint_t wid = get_group_id(0);
	uint_t wsize = get_num_groups(0);

	for (uint_t iii = SIMD * LSIZE * wid;
				iii < ni;
				iii += SIMD * LSIZE * wsize) {
		Phi_Data ip = {{0}};
		#pragma unroll SIMD
		for (uint_t i = 0, ii = iii + lid;
					i < SIMD && ii < ni;
					++i, ii += LSIZE) {
			ip._m[i] = __im[ii];
			ip._e2[i] = __ie2[ii];
			ip._rx[i] = __irdot[(0*NDIM+0)*ni + ii];
			ip._ry[i] = __irdot[(0*NDIM+1)*ni + ii];
			ip._rz[i] = __irdot[(0*NDIM+2)*ni + ii];
			ip._phi[i] = __iphi[ii];
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
				Phi_Data jp = {{0}};
				jp.m = (real_tn)(__jm[jj]);
				jp.e2 = (real_tn)(__je2[jj]);
				jp.rx = (real_tn)(__jrdot[(0*NDIM+0)*nj + jj]);
				jp.ry = (real_tn)(__jrdot[(0*NDIM+1)*nj + jj]);
				jp.rz = (real_tn)(__jrdot[(0*NDIM+2)*nj + jj]);
				jp.phi = (real_tn)(__jphi[jj]);
				barrier(CLK_LOCAL_MEM_FENCE);
				_jp[lid] = jp;
				barrier(CLK_LOCAL_MEM_FENCE);
				#pragma unroll 8
				for (uint_t j = 0; j < jlsize; ++j) {
					ip = phi_kernel_core(ip, _jp[j]);
				}
			}
		}
		#pragma unroll SIMD
		for (uint_t i = 0, ii = iii + lid;
					i < SIMD && ii < ni;
					++i, ii += LSIZE) {
			__iphi[ii] = ip._phi[i];
		}
	}
}


kernel void
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
	local Phi_Data _jp[LSIZE];

	phi_kernel_impl(
		ni, __im, __ie2, __irdot,
		nj, __jm, __je2, __jrdot,
		__iphi, __jphi,
		_jp
	);

	phi_kernel_impl(
		nj, __jm, __je2, __jrdot,
		ni, __im, __ie2, __irdot,
		__jphi, __iphi,
		_jp
	);
}


kernel void
phi_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	global real_t __iphi[])
{
	local Phi_Data _jp[LSIZE];

	phi_kernel_impl(
		ni, __im, __ie2, __irdot,
		ni, __im, __ie2, __irdot,
		__iphi, __iphi,
		_jp
	);
}

