#include <cstring>
#include "hls_stream.h"
#include "para.h"
#include "shift.h"
#include "conv2d.h"
using namespace para;
#include "weights.h"
using namespace net;

void shift(hls::stream<DataType>  tensor[C], hls::stream<DataType> act[N]){
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE axis register both port=tensor
#pragma HLS INTERFACE axis register both port=act



		static const int EXP = 1;

		static const int CHAN_0 = 16;
		static const int CHAN_1 = 32;
		static const int CHAN_2 = 64;

		static const int DIM_0 = 32;
		static const int DIM_1 = 16;
		static const int DIM_2 = 8;
		static const int DIM_3 = 1;
		static const int REX = 16;


#pragma HLS DATAFLOW
		// 3x3 conv2d
		hls::stream<DataType> s_conv[CHAN_0], s_pad[C], s_convBias[CHAN_0], s_relu[CHAN_0];
		padding<DataType,D, C, 1, REX>(tensor, s_pad);
		conv2d_3x3<DataType, D + 2, C, CHAN_0, 1, REX>(s_pad, kernel, s_conv);
		MulChan::_bias_add<DataType, D,CHAN_0, REX>(s_conv, bias0, s_convBias);
		MulChan::_relu<DataType, D,CHAN_0, REX>(s_convBias, s_relu);

		// GROUP 1
		hls::stream<DataType> s_act0[CHAN_0],s_act1[CHAN_0],s_act2[CHAN_0];
		// BLOCK 0
		MulChan::_shift<DataType, DIM_0, CHAN_0, 1, 1 * REX>(s_relu, s_act0, p0_0,p1_0, bias0_0, bias1_0);
		//BLOCK 1
		MulChan::_shift<DataType, DIM_0, CHAN_0, 1, 1 * REX>(s_act0, s_act1, p0_1,p1_1, bias0_1, bias1_1);
		//BLOCK 2
		MulChan::_shift<DataType, DIM_0, CHAN_0, 1, 1 * REX>(s_act1, s_act2, p0_2,p1_2, bias0_2, bias1_2);

		// GROUP 2
		hls::stream<DataType> s_act3[CHAN_1],s_act4[CHAN_1],s_act5[CHAN_1];
		//BLOCK 3
		MulChan::_shift_res<DataType, DIM_0, 2, CHAN_0, 1, CHAN_1, 1 * REX>(s_act2, s_act3, p0_3,p1_3, p2_3, bias0_3, bias1_3,bias2_3);
		//BLOCK 4
		MulChan::_shift<DataType, DIM_1, CHAN_1, 1, 4 * REX>(s_act3, s_act4, p0_4,p1_4, bias0_4, bias1_4);
		//BLOCK 5
		MulChan::_shift<DataType, DIM_1, CHAN_1,1, 4 * REX>(s_act4, s_act5, p0_5,p1_5, bias0_5, bias1_5);

		//GROUP 3
		hls::stream<DataType> s_act6[CHAN_2],s_act7[CHAN_2],s_act8[CHAN_2];
		//BLOCK 6
		MulChan::_shift_res<DataType, DIM_1, 2, CHAN_1, 1, CHAN_2, 4 * REX>(s_act5, s_act6, p0_6,p1_6, p2_6, bias0_6, bias1_6, bias2_6);
		//BLOCK 7
		MulChan::_shift<DataType, DIM_2, CHAN_2, 1, 16 * REX>(s_act6, s_act7, p0_7,p1_7, bias0_7, bias1_7);
		//BLOCK 8
		MulChan::_shift<DataType, DIM_2, CHAN_2,1, 16 * REX>(s_act7, s_act8, p0_8,p1_8, bias0_8, bias1_8);


		// Average pooling
		hls::stream<DataType> s_pool[CHAN_2];
		MulChan::_avg_pool<DataType, DIM_2, CHAN_2, 8, 16 * REX>(s_act8, s_pool);
		// fully-connected 
		hls::stream<DataType> s_fc[N], s_fcBias[N];
		MulChan::_matMul<DataType, DIM_3, CHAN_2, N, 16 * REX>(s_pool, s_fc, p_9);
		MulChan::_bias_add<DataType, 1, N, 16*REX>(s_fc, bias_9, act);
//		MulChan::_relu<DataType, 1, N, 16*REX>(s_fcBias, act);
}


