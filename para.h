#pragma once
#include "ap_fixed.h"
#include "hls_half.h"

#ifdef FIXED
typedef ap_fixed<32, 10> DataType;
#else
typedef half DataType;
#endif
namespace para{
	const int D = 32;
	const int C = 3;
	const int N = 10;
}
