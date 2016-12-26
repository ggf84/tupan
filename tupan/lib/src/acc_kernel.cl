#include "acc_kernel_common.h"

/*
void
acc_kernel_impl(
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
	local Acc_Data _jp[])
{
	for (uint_t ii = SIMD * get_group_id(0) * get_local_size(0);
				ii < ni;
				ii += SIMD * get_num_groups(0) * get_local_size(0)) {
		uint_t lid = get_local_id(0);
		uint_t i = ii + SIMD * lid;
		i = min(i, ni-SIMD);
		i *= (SIMD < ni);

		Acc_Data ip;
		ip.m = vec(vload)(0, __im + i);
		ip.e2 = vec(vload)(0, __ie2 + i);
		ip.rx = vec(vload)(0, &__irdot[(0*NDIM+0)*ni + i]);
		ip.ry = vec(vload)(0, &__irdot[(0*NDIM+1)*ni + i]);
		ip.rz = vec(vload)(0, &__irdot[(0*NDIM+2)*ni + i]);
		ip.ax = (real_tn)(0);
		ip.ay = (real_tn)(0);
		ip.az = (real_tn)(0);

		uint_t j = 0;

		#ifdef FAST_LOCAL_MEM
		for (; (j + LSIZE - 1) < nj; j += LSIZE) {
			Acc_Data jp;
			jp.m = (real_tn)(__jm[j + lid]);
			jp.e2 = (real_tn)(__je2[j + lid]);
			jp.rx = (real_tn)(__jrdot[(0*NDIM+0)*nj + j + lid]);
			jp.ry = (real_tn)(__jrdot[(0*NDIM+1)*nj + j + lid]);
			jp.rz = (real_tn)(__jrdot[(0*NDIM+2)*nj + j + lid]);
			jp.ax = (real_tn)(0);
			jp.ay = (real_tn)(0);
			jp.az = (real_tn)(0);
			barrier(CLK_LOCAL_MEM_FENCE);
			_jp[lid] = jp;
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll 8
			for (uint_t k = 0; k < LSIZE; ++k) {
				jp = _jp[k];
				ip = acc_kernel_core(ip, jp);
			}
		}
		#endif

		for (; j < nj; ++j) {
			Acc_Data jp;
			jp.m = (real_tn)(__jm[j]);
			jp.e2 = (real_tn)(__je2[j]);
			jp.rx = (real_tn)(__jrdot[(0*NDIM+0)*nj + j]);
			jp.ry = (real_tn)(__jrdot[(0*NDIM+1)*nj + j]);
			jp.rz = (real_tn)(__jrdot[(0*NDIM+2)*nj + j]);
			jp.ax = (real_tn)(0);
			jp.ay = (real_tn)(0);
			jp.az = (real_tn)(0);
			ip = acc_kernel_core(ip, jp);
		}

		vec(vstore)(ip.ax, 0, &__iadot[(0*NDIM+0)*ni + i]);
		vec(vstore)(ip.ay, 0, &__iadot[(0*NDIM+1)*ni + i]);
		vec(vstore)(ip.az, 0, &__iadot[(0*NDIM+2)*ni + i]);
	}
}
*/


void
acc_kernel_impl(
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
	local Acc_Data _jp[])
{
	uint_t lid = get_local_id(0);
	uint_t gid = LSIZE * get_group_id(0) + lid;
	uint_t gsize = LSIZE * get_num_groups(0);

	for (uint_t ii = gid;
				ii < ni;
				ii += gsize) {
		__iadot[(0*NDIM+0)*ni + ii] = 0;
		__iadot[(0*NDIM+1)*ni + ii] = 0;
		__iadot[(0*NDIM+2)*ni + ii] = 0;
	}

	for (uint_t ii = gid;
				ii < ni;
				ii += gsize) {
		Acc_Data ip;
		ip.m = (real_tn)(__im[ii]);
		ip.e2 = (real_tn)(__ie2[ii]);
		ip.rx = (real_tn)(__irdot[(0*NDIM+0)*ni + ii]);
		ip.ry = (real_tn)(__irdot[(0*NDIM+1)*ni + ii]);
		ip.rz = (real_tn)(__irdot[(0*NDIM+2)*ni + ii]);
		ip.ax = (real_tn)(__iadot[(0*NDIM+0)*ni + ii]);
		ip.ay = (real_tn)(__iadot[(0*NDIM+1)*ni + ii]);
		ip.az = (real_tn)(__iadot[(0*NDIM+2)*ni + ii]);
		for (uint_t jj = LSIZE * (nj/LSIZE);
					jj + 1 - 1 < nj;
					jj += 1) {
			Acc_Data jp;
			jp.m = (real_tn)(__jm[jj]);
			jp.e2 = (real_tn)(__je2[jj]);
			jp.rx = (real_tn)(__jrdot[(0*NDIM+0)*nj + jj]);
			jp.ry = (real_tn)(__jrdot[(0*NDIM+1)*nj + jj]);
			jp.rz = (real_tn)(__jrdot[(0*NDIM+2)*nj + jj]);
			jp.ax = (real_tn)(__jadot[(0*NDIM+0)*nj + jj]);
			jp.ay = (real_tn)(__jadot[(0*NDIM+1)*nj + jj]);
			jp.az = (real_tn)(__jadot[(0*NDIM+2)*nj + jj]);
			ip = acc_kernel_core(ip, jp);
		}
		__iadot[(0*NDIM+0)*ni + ii] = ip.ax;
		__iadot[(0*NDIM+1)*ni + ii] = ip.ay;
		__iadot[(0*NDIM+2)*ni + ii] = ip.az;
	}

	for (uint_t jblock = 0;
				jblock + LSIZE - 1 < nj;
				jblock += LSIZE) {
		Acc_Data jp;
		uint_t jj = jblock + lid;
		jp.m = (real_tn)(__jm[jj]);
		jp.e2 = (real_tn)(__je2[jj]);
		jp.rx = (real_tn)(__jrdot[(0*NDIM+0)*nj + jj]);
		jp.ry = (real_tn)(__jrdot[(0*NDIM+1)*nj + jj]);
		jp.rz = (real_tn)(__jrdot[(0*NDIM+2)*nj + jj]);
		jp.ax = (real_tn)(__jadot[(0*NDIM+0)*nj + jj]);
		jp.ay = (real_tn)(__jadot[(0*NDIM+1)*nj + jj]);
		jp.az = (real_tn)(__jadot[(0*NDIM+2)*nj + jj]);
		barrier(CLK_LOCAL_MEM_FENCE);
		_jp[lid] = jp;
		barrier(CLK_LOCAL_MEM_FENCE);
		for (uint_t ii = gid;
					ii < ni;
					ii += gsize) {
			Acc_Data ip;
			ip.m = (real_tn)(__im[ii]);
			ip.e2 = (real_tn)(__ie2[ii]);
			ip.rx = (real_tn)(__irdot[(0*NDIM+0)*ni + ii]);
			ip.ry = (real_tn)(__irdot[(0*NDIM+1)*ni + ii]);
			ip.rz = (real_tn)(__irdot[(0*NDIM+2)*ni + ii]);
			ip.ax = (real_tn)(__iadot[(0*NDIM+0)*ni + ii]);
			ip.ay = (real_tn)(__iadot[(0*NDIM+1)*ni + ii]);
			ip.az = (real_tn)(__iadot[(0*NDIM+2)*ni + ii]);

			#pragma unroll 8
			for (uint_t j = 0; j < LSIZE; ++j) {
				ip = acc_kernel_core(ip, _jp[j]);
			}

			__iadot[(0*NDIM+0)*ni + ii] = ip.ax;
			__iadot[(0*NDIM+1)*ni + ii] = ip.ay;
			__iadot[(0*NDIM+2)*ni + ii] = ip.az;
		}
	}
}


/*
void
acc_kernel_impl(
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
	local Acc_Data _jp[])
{
	uint_t lid = get_local_id(0);
	uint_t gid = LSIZE * get_group_id(0) + lid;
	uint_t gsize = LSIZE * get_num_groups(0);
	uint_t iend = (ni + SIMD - 1) / SIMD;
	uint_t nsimd = (ni - SIMD) * (SIMD < ni);

	for (uint_t iii = gid;
				iii < iend;
				iii += gsize) {
		uint_t ii = min(iii * SIMD, nsimd);
		vec(vstore)((real_tn)(0), 0, &__iadot[(0*NDIM+0)*ni + ii]);
		vec(vstore)((real_tn)(0), 0, &__iadot[(0*NDIM+1)*ni + ii]);
		vec(vstore)((real_tn)(0), 0, &__iadot[(0*NDIM+2)*ni + ii]);
	}

	for (uint_t iii = gid;
				iii < iend;
				iii += gsize) {
		Acc_Data ip;
		uint_t ii = min(iii * SIMD, nsimd);
		ip.m = vec(vload)(0, &__im[ii]);
		ip.e2 = vec(vload)(0, &__ie2[ii]);
		ip.rx = vec(vload)(0, &__irdot[(0*NDIM+0)*ni + ii]);
		ip.ry = vec(vload)(0, &__irdot[(0*NDIM+1)*ni + ii]);
		ip.rz = vec(vload)(0, &__irdot[(0*NDIM+2)*ni + ii]);
		ip.ax = vec(vload)(0, &__iadot[(0*NDIM+0)*ni + ii]);
		ip.ay = vec(vload)(0, &__iadot[(0*NDIM+1)*ni + ii]);
		ip.az = vec(vload)(0, &__iadot[(0*NDIM+2)*ni + ii]);
		for (uint_t jj = LSIZE * (nj/LSIZE);
					jj + 1 - 1 < nj;
					jj += 1) {
			Acc_Data jp;
			jp.m = (real_tn)(__jm[jj]);
			jp.e2 = (real_tn)(__je2[jj]);
			jp.rx = (real_tn)(__jrdot[(0*NDIM+0)*nj + jj]);
			jp.ry = (real_tn)(__jrdot[(0*NDIM+1)*nj + jj]);
			jp.rz = (real_tn)(__jrdot[(0*NDIM+2)*nj + jj]);
			jp.ax = (real_tn)(__jadot[(0*NDIM+0)*nj + jj]);
			jp.ay = (real_tn)(__jadot[(0*NDIM+1)*nj + jj]);
			jp.az = (real_tn)(__jadot[(0*NDIM+2)*nj + jj]);
			ip = acc_kernel_core(ip, jp);
		}
		vec(vstore)(ip.ax, 0, &__iadot[(0*NDIM+0)*ni + ii]);
		vec(vstore)(ip.ay, 0, &__iadot[(0*NDIM+1)*ni + ii]);
		vec(vstore)(ip.az, 0, &__iadot[(0*NDIM+2)*ni + ii]);
	}

	for (uint_t jblock = 0;
				jblock + LSIZE - 1 < nj;
				jblock += LSIZE) {
		Acc_Data jp;
		uint_t jj = jblock + lid;
		jp.m = (real_tn)(__jm[jj]);
		jp.e2 = (real_tn)(__je2[jj]);
		jp.rx = (real_tn)(__jrdot[(0*NDIM+0)*nj + jj]);
		jp.ry = (real_tn)(__jrdot[(0*NDIM+1)*nj + jj]);
		jp.rz = (real_tn)(__jrdot[(0*NDIM+2)*nj + jj]);
		jp.ax = (real_tn)(__jadot[(0*NDIM+0)*nj + jj]);
		jp.ay = (real_tn)(__jadot[(0*NDIM+1)*nj + jj]);
		jp.az = (real_tn)(__jadot[(0*NDIM+2)*nj + jj]);
		barrier(CLK_LOCAL_MEM_FENCE);
		_jp[lid] = jp;
		barrier(CLK_LOCAL_MEM_FENCE);
		for (uint_t iii = gid;
					iii < iend;
					iii += gsize) {
			Acc_Data ip;
			uint_t ii = min(iii * SIMD, nsimd);
			ip.m = vec(vload)(0, &__im[ii]);
			ip.e2 = vec(vload)(0, &__ie2[ii]);
			ip.rx = vec(vload)(0, &__irdot[(0*NDIM+0)*ni + ii]);
			ip.ry = vec(vload)(0, &__irdot[(0*NDIM+1)*ni + ii]);
			ip.rz = vec(vload)(0, &__irdot[(0*NDIM+2)*ni + ii]);
			ip.ax = vec(vload)(0, &__iadot[(0*NDIM+0)*ni + ii]);
			ip.ay = vec(vload)(0, &__iadot[(0*NDIM+1)*ni + ii]);
			ip.az = vec(vload)(0, &__iadot[(0*NDIM+2)*ni + ii]);

			#pragma unroll 8
			for (uint_t j = 0; j < LSIZE; ++j) {
				ip = acc_kernel_core(ip, _jp[j]);
			}

			vec(vstore)(ip.ax, 0, &__iadot[(0*NDIM+0)*ni + ii]);
			vec(vstore)(ip.ay, 0, &__iadot[(0*NDIM+1)*ni + ii]);
			vec(vstore)(ip.az, 0, &__iadot[(0*NDIM+2)*ni + ii]);
		}
	}
}
*/

/*
void
acc_kernel_impl(
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
	local Acc_Data _ip[])
{
	uint_t lid = get_local_id(0);
	for (uint_t iii = SIMD * LSIZE * get_group_id(0);
				iii < ni;
				iii += SIMD * LSIZE * get_num_groups(0)) {
		Acc_Data ip;
		uint_t ii = min(iii + SIMD * lid, (ni - SIMD) * (SIMD < ni));
		ip.m = vec(vload)(0, &__im[ii]);
		ip.e2 = vec(vload)(0, &__ie2[ii]);
		ip.rx = vec(vload)(0, &__irdot[(0*NDIM+0)*ni + ii]);
		ip.ry = vec(vload)(0, &__irdot[(0*NDIM+1)*ni + ii]);
		ip.rz = vec(vload)(0, &__irdot[(0*NDIM+2)*ni + ii]);
		ip.ax = (real_tn)(0);//vec(vload)(0, &__iadot[(0*NDIM+0)*ni + ii]);
		ip.ay = (real_tn)(0);//vec(vload)(0, &__iadot[(0*NDIM+1)*ni + ii]);
		ip.az = (real_tn)(0);//vec(vload)(0, &__iadot[(0*NDIM+2)*ni + ii]);
		_ip[lid] = ip;

		barrier(CLK_LOCAL_MEM_FENCE);
		for (uint_t jj = lid;
					jj < nj;
					jj += LSIZE) {
			Acc_Data jp;
			jp.m = (real_tn)(__jm[jj]);
			jp.e2 = (real_tn)(__je2[jj]);
			jp.rx = (real_tn)(__jrdot[(0*NDIM+0)*nj + jj]);
			jp.ry = (real_tn)(__jrdot[(0*NDIM+1)*nj + jj]);
			jp.rz = (real_tn)(__jrdot[(0*NDIM+2)*nj + jj]);
			jp.ax = (real_tn)(0);//(real_tn)(__jadot[(0*NDIM+0)*nj + jj]);
			jp.ay = (real_tn)(0);//(real_tn)(__jadot[(0*NDIM+1)*nj + jj]);
			jp.az = (real_tn)(0);//(real_tn)(__jadot[(0*NDIM+2)*nj + jj]);

			#pragma unroll 8
			for (uint_t k = 0; k < LSIZE; ++k) {
				_ip[lid^k] = acc_kernel_core(_ip[lid^k], jp);
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		ip = _ip[lid];
		vec(vstore)(ip.ax, 0, &__iadot[(0*NDIM+0)*ni + ii]);
		vec(vstore)(ip.ay, 0, &__iadot[(0*NDIM+1)*ni + ii]);
		vec(vstore)(ip.az, 0, &__iadot[(0*NDIM+2)*ni + ii]);
	}
}
*/


kernel void
acc_kernel_rectangle(
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
	local Acc_Data _pAcc[LSIZE];

	acc_kernel_impl(
		ni, __im, __ie2, __irdot,
		nj, __jm, __je2, __jrdot,
		__iadot, __jadot,
		_pAcc
	);

	barrier(CLK_GLOBAL_MEM_FENCE);

	acc_kernel_impl(
		nj, __jm, __je2, __jrdot,
		ni, __im, __ie2, __irdot,
		__jadot, __iadot,
		_pAcc
	);
}


kernel void
acc_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	global real_t __iadot[])
{
	local Acc_Data _pAcc[LSIZE];

	acc_kernel_impl(
		ni, __im, __ie2, __irdot,
		ni, __im, __ie2, __irdot,
		__iadot, __iadot,
		_pAcc
	);
}

