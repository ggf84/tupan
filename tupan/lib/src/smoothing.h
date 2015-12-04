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
get_inv_r(
	const real_tn r2,
	const real_tn e2,
	const int_tn mask)
// flop count: 3
{
	real_tn inv_r = rsqrt(r2 + e2);
	return select((real_tn)(0), inv_r, mask);
}


static inline real_tn
smoothed_m_r1(
	const real_tn m,
	const real_tn r2,
	const real_tn e2,
	const int_tn mask)
// flop count: 4
{
	real_tn inv_r = get_inv_r(r2, e2, mask);
	return m * inv_r;
}


static inline real_tn
smoothed_m_r2(
	const real_tn m,
	const real_tn r2,
	const real_tn e2,
	const int_tn mask)
// flop count: 5
{
	real_tn inv_r = get_inv_r(r2, e2, mask);
	return m * inv_r * inv_r;
}


static inline real_tn
smoothed_m_r3(
	const real_tn m,
	const real_tn r2,
	const real_tn e2,
	const int_tn mask)
// flop count: 6
{
	real_tn inv_r = get_inv_r(r2, e2, mask);
	return m * inv_r * inv_r * inv_r;
}


static inline real_tn
smoothed_m_r1_inv_r2(
	const real_tn m,
	const real_tn r2,
	const real_tn e2,
	const int_tn mask,
	real_tn *inv_r2)
// flop count: 5
{
	real_tn inv_r = get_inv_r(r2, e2, mask);
	*inv_r2 = inv_r * inv_r;
	return m * inv_r;
}


static inline real_tn
smoothed_m_r3_inv_r2(
	const real_tn m,
	const real_tn r2,
	const real_tn e2,
	const int_tn mask,
	real_tn *inv_r2)
// flop count: 6
{
	real_tn inv_r = get_inv_r(r2, e2, mask);
	*inv_r2 = inv_r * inv_r;
	return m * inv_r * *inv_r2;
}


static inline real_tn
smoothed_m_r1_m_r3(
	const real_tn m,
	const real_tn r2,
	const real_tn e2,
	const int_tn mask,
	real_tn *m_r3)
// flop count: 6
{
	real_tn inv_r = get_inv_r(r2, e2, mask);
	real_tn m_r1 = m * inv_r;
	*m_r3 = inv_r * inv_r * m_r1;
	return m_r1;
}


static inline real_tn
smoothed_inv_r1r2(
	const real_tn r2,
	const real_tn e2,
	const int_tn mask,
	real_tn *inv_r2)
// flop count: 4
{
	real_tn inv_r = get_inv_r(r2, e2, mask);
	*inv_r2 = inv_r * inv_r;
	return inv_r;
}


#endif	// __SMOOTHING_H__
