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
		hls::stream<DataType> f_conv0, f_shift,f_conv1, f_pool;
#pragma HLS STREAM variable=f_shift depth=1 dim=1
#pragma HLS STREAM variable=f_conv0 depth=1 dim=1
#pragma HLS STREAM variable=f_pool depth=1 dim=1
#pragma HLS STREAM variable=f_conv1 depth=1 dim=1

		_linear_combination<DataType,D, C, M>(tensor, f_conv0, p0);
		_shift_3x3<DataType, D, M, S>(f_conv0, f_shift, Dx, Dy);
		_linear_combination<DataType,nD, M, N>(f_shift, f_conv1, p1);
		_max_pool<DataType, nD, N, 2>(f_conv1,f_pool);
		_relu<DataType, nD/2, N>(f_pool, act);
}



