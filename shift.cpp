#include <cstring>
#include "hls_stream.h"
#include "para.h"
#include "shift.h"
#include "conv2d.h"
using namespace para;


#ifdef SINGLE
void shift(hls::stream<DataType> & tensor,
		hls::stream<DataType> & act){
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE axis register both port=tensor
#pragma HLS INTERFACE axis register both port=act


#pragma HLS DATAFLOW
		static const int CHAN_0 = 16;
		static const int CHAN_1 = 32;
		static const int CHAN_2 = 64;

		static const int DIM_0 = 32;
		static const int DIM_1 = 16;
		static const int DIM_2 = 8;


#pragma HLS DATAFLOW

		// GROUP 1
		hls::stream<DataType> s_act0,s_act1,s_act2;
		// BLOCK 0
		const int D0[C]={
#include "d_0"
		};
		const DataType p0_0[C][C]={
#include "p0_0"
		};
		const DataType p1_0[C][CHAN_0]={
#include "p1_0"
		};
		SingleChan::_shift<DataType, DIM_0, 1, 1, C, 1, CHAN_0>(tensor, s_act0, D0 ,p0_0,p1_0);

		//BLOCK 1
		const int D1[CHAN_0]={
#include "d_1"
		};
		const DataType p0_1[CHAN_0][CHAN_0]={
#include "p0_1"
		};
		const DataType p1_1[CHAN_0][CHAN_0]={
#include "p1_1"
		};
		SingleChan::_shift<DataType, DIM_0, 1, 1, CHAN_0, 1, CHAN_0>(s_act0, s_act1, D1 ,p0_1,p1_1);

		//BLOCK 2
		const int D2[CHAN_0]={
#include "d_2"
		};
		const DataType p0_2[CHAN_0][CHAN_0]={
#include "p0_2"
		};
		const DataType p1_2[CHAN_0][CHAN_0]={
#include "p1_2"
		};
		SingleChan::_shift<DataType, DIM_0, 1, 1, CHAN_0, 1, CHAN_0>(s_act1, act, D2 ,p0_2,p1_2);

}
#elif defined SINGLE_NET

void shift(hls::stream<DataType>  &tensor, hls::stream<DataType> &act){
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE axis register both port=tensor
#pragma HLS INTERFACE axis register both port=act



		static const int CHAN_0 = 16;
		static const int CHAN_1 = 32;
		static const int CHAN_2 = 64;

		static const int DIM_0 = 32;
		static const int DIM_1 = 16;
		static const int DIM_2 = 8;


		const DataType kernel[3][3][C][CHAN_0]={
#include "t_k"
		};

		hls::stream<DataType> s_pad, s_conv;
		padding<DataType, DIM_0, C, 1>(tensor, s_pad);	
		conv2d<DataType, DIM_0 + 2, C, 3, CHAN_0, 1>(s_pad, kernel, s_conv);
#pragma HLS DATAFLOW

		// GROUP 1
		hls::stream<DataType> s_act0,s_act1,s_act2;
		// BLOCK 0
		const int D0[CHAN_0]={
#include "d_0"
		};
		const DataType p0_0[CHAN_0][CHAN_0]={
#include "p0_0"
		};
		const DataType p1_0[CHAN_0][CHAN_0]={
#include "p1_0"
		};
		SingleChan::_shift<DataType, DIM_0, 1, 1, CHAN_0, 1, CHAN_0>(s_conv, s_act0, D0 ,p0_0,p1_0);

		//BLOCK 1
		const int D1[CHAN_0]={
#include "d_1"
		};
		const DataType p0_1[CHAN_0][CHAN_0]={
#include "p0_1"
		};
		const DataType p1_1[CHAN_0][CHAN_0]={
#include "p1_1"
		};
		SingleChan::_shift<DataType, DIM_0, 1, 1, CHAN_0, 1, CHAN_0>(s_act0, s_act1, D1 ,p0_1,p1_1);

		//BLOCK 2
		const int D2[CHAN_0]={
#include "d_2"
		};
		const DataType p0_2[CHAN_0][CHAN_0]={
#include "p0_2"
		};
		const DataType p1_2[CHAN_0][CHAN_0]={
#include "p1_2"
		};
		SingleChan::_shift<DataType, DIM_0, 1, 1, CHAN_0, 1, CHAN_0>(s_act1, s_act2, D2 ,p0_2,p1_2);

		// GROUP 2
		hls::stream<DataType> s_act3,s_act4,s_act5;
		//BLOCK 3
		const int D3[CHAN_0]={
#include "d_3"
		};
		const DataType p0_3[CHAN_0][CHAN_0]={
#include "p0_3"
		};
		const DataType p1_3[CHAN_0][CHAN_1]={
#include "p1_3"
		};
		SingleChan::_shift<DataType, DIM_0, 1, 2, CHAN_0, 1, CHAN_1>(s_act2, s_act3, D3 ,p0_3,p1_3);
		//BLOCK 4
		const int D4[CHAN_1]={
#include "d_4"
		};
		const DataType p0_4[CHAN_1][CHAN_1]={
#include "p0_4"
		};
		const DataType p1_4[CHAN_1][CHAN_1]={
#include "p1_4"
		};
		SingleChan::_shift<DataType, DIM_1, 1, 1, CHAN_1, 1, CHAN_1>(s_act3, s_act4, D4 ,p0_4,p1_4);
		//BLOCK 5
		const int D5[CHAN_1]={
#include "d_5"
		};
		const DataType p0_5[CHAN_1][CHAN_1]={
#include "p0_5"
		};
		const DataType p1_5[CHAN_1][CHAN_1]={
#include "p1_5"
		};
		SingleChan::_shift<DataType, DIM_1, 1, 1, CHAN_1, 1, CHAN_1>(s_act4, s_act5, D5 ,p0_5,p1_5);
		//GROUP 3
		hls::stream<DataType> s_act6,s_act7,s_act8;
		//BLOCK 6
		const int D6[CHAN_1]={
#include "d_6"
		};
		const DataType p0_6[CHAN_1][CHAN_1]={
#include "p0_6"
		};
		const DataType p1_6[CHAN_1][CHAN_2]={
#include "p1_6"
		};
		SingleChan::_shift<DataType, DIM_1, 1, 2, CHAN_1, 1, CHAN_2>(s_act5, s_act6, D6 ,p0_6,p1_6);
		//BLOCK 7
		const int D7[CHAN_2]={
#include "d_7"
		};
		const DataType p0_7[CHAN_2][CHAN_2]={
#include "p0_7"
		};
		const DataType p1_7[CHAN_2][CHAN_2]={
#include "p1_7"
		};
		SingleChan::_shift<DataType, DIM_2, 1, 1, CHAN_2, 1, CHAN_2>(s_act6, s_act7, D7 ,p0_7,p1_7);
		//BLOCK 8
		const int D8[CHAN_2]={
#include "d_8"
		};
		const DataType p0_8[CHAN_2][CHAN_2]={
#include "p0_8"
		};
		const DataType p1_8[CHAN_2][CHAN_2]={
#include "p1_8"
		};
		SingleChan::_shift<DataType, DIM_2, 1, 1, CHAN_2, 1, CHAN_2>(s_act7, s_act8, D8 ,p0_8,p1_8);


		const DataType p_9[DIM_2 * DIM_2* CHAN_2][N]={
#include "p_9"
		};

		hls::stream<DataType> s_fc;
		SingleChan::_matMul<DataType, DIM_2, CHAN_2, N>(s_act8, s_fc, p_9);
		SingleChan::_relu<DataType, 1, N>(s_fc, act);
}

#endif

