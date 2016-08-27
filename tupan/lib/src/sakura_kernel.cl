#include "sakura_kernel_common.h"


kernel void
sakura_kernel(
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
	global real_t __idrdot[])
{
	uint_t lid = get_local_id(0);
	uint_t start = get_group_id(0) * get_local_size(0);
	uint_t stride = get_num_groups(0) * get_local_size(0);
	for (uint_t ii = start; ii * 1 < ni; ii += stride) {
		uint_t i = ii + lid;
		i *= 1;
		i = (i+1 < ni) ? (i):(ni-1);
		i *= (1 < ni);

		Sakura_Data1 ip;
		ip.m = vload1(0, __im + i);
		ip.e2 = vload1(0, __ie2 + i);
		#pragma unroll
		for (uint_t kdot = 0; kdot < 2; ++kdot) {
			#pragma unroll
			for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
				global const real_t *ptr = &__irdot[(kdot*NDIM+kdim)*ni];
				ip.rdot[kdot][kdim] = vload1(0, ptr + i);
				ip.drdot[kdot][kdim] = (real_t)(0);
			}
		}

		uint_t j = 0;

		#ifdef FAST_LOCAL_MEM
		for (; ((j + LSIZE) - 1) < nj; j += LSIZE) {
			Sakura_Data jp;
			jp.m = __jm[j + lid];
			jp.e2 = __je2[j + lid];
			#pragma unroll
			for (uint_t kdot = 0; kdot < 2; ++kdot) {
				#pragma unroll
				for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
					global const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
					jp.rdot[kdot][kdim] = ptr[j + lid];
					jp.drdot[kdot][kdim] = 0;
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			local Sakura_Data _jp[LSIZE];
			_jp[lid] = jp;
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll
			for (uint_t k = 0; k < LSIZE; ++k) {
				ip = sakura_kernel_core(ip, _jp[k], dt, flag);
			}
		}
		#endif

		for (; ((j + 1) - 1) < nj; j += 1) {
			Sakura_Data jp;
			jp.m = __jm[j];
			jp.e2 = __je2[j];
			#pragma unroll
			for (uint_t kdot = 0; kdot < 2; ++kdot) {
				#pragma unroll
				for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
					global const real_t *ptr = &__jrdot[(kdot*NDIM+kdim)*nj];
					jp.rdot[kdot][kdim] = ptr[j];
					jp.drdot[kdot][kdim] = 0;
				}
			}
			ip = sakura_kernel_core(ip, jp, dt, flag);
		}

		#pragma unroll
		for (uint_t kdot = 0; kdot < 2; ++kdot) {
			#pragma unroll
			for (uint_t kdim = 0; kdim < NDIM; ++kdim) {
				global real_t *ptr = &__idrdot[(kdot*NDIM+kdim)*ni];
				vstore1(ip.drdot[kdot][kdim], 0, ptr + i);
			}
		}
	}
}

