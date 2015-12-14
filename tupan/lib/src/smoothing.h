#ifndef __SMOOTHING_H__
#define __SMOOTHING_H__

#include "common.h"


static inline real_tn
smoothed_inv_r1(
	const real_tn r2,
	const real_tn e2)
// flop count: 3
{
	return rsqrt(r2 + e2);
}


static inline real_tn
smoothed_inv_r2(
	const real_tn r2,
	const real_tn e2)
// flop count: 4
{
	real_tn inv_r = rsqrt(r2 + e2);
	return inv_r * inv_r;
}


static inline real_tn
smoothed_inv_r3(
	const real_tn r2,
	const real_tn e2)
// flop count: 5
{
	real_tn inv_r = rsqrt(r2 + e2);
	return inv_r * inv_r * inv_r;
}


static inline real_tn
smoothed_inv_r2_inv_r1(
	const real_tn r2,
	const real_tn e2,
	real_tn *inv_r1)
// flop count: 4
{
	real_tn inv_r = rsqrt(r2 + e2);
	*inv_r1 = inv_r;
	return inv_r * inv_r;
}


static inline real_tn
smoothed_inv_r3_inv_r1(
	const real_tn r2,
	const real_tn e2,
	real_tn *inv_r1)
// flop count: 5
{
	real_tn inv_r = rsqrt(r2 + e2);
	*inv_r1 = inv_r;
	return inv_r * inv_r * inv_r;
}


static inline real_tn
smoothed_inv_r3_inv_r2(
	const real_tn r2,
	const real_tn e2,
	real_tn *inv_r2)
// flop count: 5
{
	real_tn inv_r = rsqrt(r2 + e2);
	*inv_r2 = inv_r * inv_r;
	return inv_r * *inv_r2;
}


#endif	// __SMOOTHING_H__
