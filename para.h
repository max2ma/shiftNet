#pragma once
#include "ap_fixed.h"
#include "hls_half.h"

//#define MUL
#define SINGLE_NET
//#define FIXED

#ifdef FIXED
typedef ap_fixed<16, 4> DataType;
#else
typedef half DataType;
#endif
namespace para{
	const int D = 32;
#if defined SINGLE_NET || defined MUL_NET
	const int C = 3;
	const int N = 10;
#else
	const int C = 16;
	const int N = 32;
#endif
	const int E = 1;
	const int M = E*C;
	const int sS = 1;
	const int cS = 1;
	const int mS = 2;
	const int nD = (D - 1)/sS/cS + 1;
}
