#pragma once
#include "ap_fixed.h"
#include "hls_half.h"


#define MUL_NET
//#define FIXED

#ifdef FIXED
typedef ap_fixed<16, 4> DataType;
#else
typedef half DataType;
#endif
namespace para{
	const int D = 32;
	const int C = 3;
	const int E = 1;
	const int M = E*C;
	const int N = 10;
	const int sS = 1;
	const int cS = 1;
	const int mS = 2;
	const int nD = (D - 1)/sS/cS + 1;
}
