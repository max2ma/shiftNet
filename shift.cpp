#include "hls_stream.h"
#include "para.h"
#include "shift.h"
using namespace para;


void shift(hls::stream<DataType> & tensor,
		hls::stream<DataType> & act){
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE axis register both port=tensor
#pragma HLS INTERFACE axis register both port=act

		const int Dx[M]={
#include "dx"
		};
		const int Dy[M]={
#include "dy"
		};
		const DataType p0[C][M]={
#include "p0"
		};
		const DataType p1[M][N]={
#include "p1"
		};

#pragma HLS DATAFLOW
		_shift<DataType, D, sS, cS, mS, C, M, N>(tensor, act, Dx, Dy,p0,p1);
}



