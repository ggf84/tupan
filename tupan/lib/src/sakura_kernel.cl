#include "sakura_kernel_common.h"


void
sakura_kernel_impl(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	const real_t dt,
	const int_t flag,
	global real_t __idrdot[],
	global real_t __jdrdot[],
	local Sakura_Data _jp[])
{
	uint_t lid = get_local_id(0);
	uint_t wid = get_group_id(0);
	uint_t wsize = get_num_groups(0);

	for (uint_t iii = 1 * LSIZE * wid;
				iii < ni;
				iii += 1 * LSIZE * wsize) {
		Sakura_Data ip = {{0}};
//		#pragma unroll 1
		for (uint_t i = 0, ii = iii + lid;
					i < 1 && ii < ni;
					++i, ii += LSIZE) {
			ip._m[i] = __im[ii];
			ip._e2[i] = __ie2[ii];
			ip._rx[i] = __irdot[(0*NDIM+0)*ni + ii];
			ip._ry[i] = __irdot[(0*NDIM+1)*ni + ii];
			ip._rz[i] = __irdot[(0*NDIM+2)*ni + ii];
			ip._vx[i] = __irdot[(1*NDIM+0)*ni + ii];
			ip._vy[i] = __irdot[(1*NDIM+1)*ni + ii];
			ip._vz[i] = __irdot[(1*NDIM+2)*ni + ii];
			ip._drx[i] = __idrdot[(0*NDIM+0)*ni + ii];
			ip._dry[i] = __idrdot[(0*NDIM+1)*ni + ii];
			ip._drz[i] = __idrdot[(0*NDIM+2)*ni + ii];
			ip._dvx[i] = __idrdot[(1*NDIM+0)*ni + ii];
			ip._dvy[i] = __idrdot[(1*NDIM+1)*ni + ii];
			ip._dvz[i] = __idrdot[(1*NDIM+2)*ni + ii];
		}
		uint_t j0 = 0;
		uint_t j1 = 0;
//		#pragma unroll
		for (uint_t jlsize = LSIZE;
					jlsize > 0;
					jlsize >>= 1) {
			j0 = j1 + lid % jlsize;
			j1 = jlsize * (nj/jlsize);
			for (uint_t jj = j0;
						jj < j1;
						jj += jlsize) {
				Sakura_Data jp = {{0}};
				jp.m = (real_t1)(__jm[jj]);
				jp.e2 = (real_t1)(__je2[jj]);
				jp.rx = (real_t1)(__jrdot[(0*NDIM+0)*nj + jj]);
				jp.ry = (real_t1)(__jrdot[(0*NDIM+1)*nj + jj]);
				jp.rz = (real_t1)(__jrdot[(0*NDIM+2)*nj + jj]);
				jp.vx = (real_t1)(__jrdot[(1*NDIM+0)*nj + jj]);
				jp.vy = (real_t1)(__jrdot[(1*NDIM+1)*nj + jj]);
				jp.vz = (real_t1)(__jrdot[(1*NDIM+2)*nj + jj]);
				jp.drx = (real_t1)(__jdrdot[(0*NDIM+0)*nj + jj]);
				jp.dry = (real_t1)(__jdrdot[(0*NDIM+1)*nj + jj]);
				jp.drz = (real_t1)(__jdrdot[(0*NDIM+2)*nj + jj]);
				jp.dvx = (real_t1)(__jdrdot[(1*NDIM+0)*nj + jj]);
				jp.dvy = (real_t1)(__jdrdot[(1*NDIM+1)*nj + jj]);
				jp.dvz = (real_t1)(__jdrdot[(1*NDIM+2)*nj + jj]);
				barrier(CLK_LOCAL_MEM_FENCE);
				_jp[lid] = jp;
				barrier(CLK_LOCAL_MEM_FENCE);
//				#pragma unroll 8
				for (uint_t j = 0; j < jlsize; ++j) {
					ip = sakura_kernel_core(ip, _jp[j], dt, flag);
				}
			}
		}
//		#pragma unroll 1
		for (uint_t i = 0, ii = iii + lid;
					i < 1 && ii < ni;
					++i, ii += LSIZE) {
			__idrdot[(0*NDIM+0)*ni + ii] = ip._drx[i];
			__idrdot[(0*NDIM+1)*ni + ii] = ip._dry[i];
			__idrdot[(0*NDIM+2)*ni + ii] = ip._drz[i];
			__idrdot[(1*NDIM+0)*ni + ii] = ip._dvx[i];
			__idrdot[(1*NDIM+1)*ni + ii] = ip._dvy[i];
			__idrdot[(1*NDIM+2)*ni + ii] = ip._dvz[i];
		}
	}
}


kernel void
sakura_kernel_rectangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	const real_t dt,
	const int_t flag,
	global real_t __idrdot[],
	global real_t __jdrdot[])
{
	local Sakura_Data _jp[LSIZE];

	sakura_kernel_impl(
		ni, __im, __ie2, __irdot,
		nj, __jm, __je2, __jrdot,
		dt, flag,
		__idrdot, __jdrdot,
		_jp
	);

	sakura_kernel_impl(
		nj, __jm, __je2, __jrdot,
		ni, __im, __ie2, __irdot,
		dt, flag,
		__jdrdot, __idrdot,
		_jp
	);
}


kernel void
sakura_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const real_t dt,
	const int_t flag,
	global real_t __idrdot[])
{
	local Sakura_Data _jp[LSIZE];

	sakura_kernel_impl(
		ni, __im, __ie2, __irdot,
		ni, __im, __ie2, __irdot,
		dt, flag,
		__idrdot, __idrdot,
		_jp
	);
}

