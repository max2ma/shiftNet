#pragma once
#include "para.h"
using namespace para;
namespace net{
	static const int EXP = 1;

	static const int CHAN_0 = 16;
	static const int CHAN_1 = 32;
	static const int CHAN_2 = 64;

	static const int DIM_0 = 32;
	static const int DIM_1 = 16;
	static const int DIM_2 = 8;
	static const int REX = 16;

	const DataType kernel[3][3][C][CHAN_0] = {
#include "t_k"
	};
	const DataType bias0[CHAN_0]={
#include "bias0"
	};
	//BLOCK 0
	const DataType p0_0[CHAN_0][CHAN_0 * EXP]={
#include "p0_0"
	};
	const DataType p1_0[CHAN_0 * EXP][CHAN_0]={
#include "p1_0"
	};
	const DataType bias0_0[CHAN_0 * EXP]={
#include "bias0_0"
	};
	const DataType bias1_0[CHAN_0]={
#include "bias1_0"
	};

	//BLOCK 1
	const DataType p0_1[CHAN_0][CHAN_0 * EXP]={
#include "p0_1"
	};
	const DataType p1_1[CHAN_0 * EXP][CHAN_0]={
#include "p1_1"
	};
	const DataType bias0_1[CHAN_0 * EXP]={
#include "bias0_1"
	};
	const DataType bias1_1[CHAN_0]={
#include "bias1_1"
	};

	//BLOCK 2
	const DataType p0_2[CHAN_0][CHAN_0]={
#include "p0_2"
	};
	const DataType p1_2[CHAN_0][CHAN_0]={
#include "p1_2"
	};
	const DataType bias0_2[CHAN_0 * EXP]={
#include "bias0_2"
	};
	const DataType bias1_2[CHAN_0]={
#include "bias1_2"
	};

	// GROUP 2
	//BLOCK 3
	const DataType p0_3[CHAN_0][CHAN_1]={
#include "p0_3"
	};
	const DataType p1_3[CHAN_1][CHAN_1]={
#include "p1_3"
	};
	const DataType p2_3[CHAN_0][CHAN_1]={
#include "p2_3"
	};
	const DataType bias0_3[CHAN_1 * EXP]={
#include "bias0_3"
	};
	const DataType bias1_3[CHAN_1]={
#include "bias1_3"
	};
	const DataType bias2_3[CHAN_1]={
#include "bias2_3"
	};
	//BLOCK 4
	const DataType p0_4[CHAN_1][CHAN_1]={
#include "p0_4"
	};
	const DataType p1_4[CHAN_1][CHAN_1]={
#include "p1_4"
	};
	const DataType bias0_4[CHAN_1 * EXP]={
#include "bias0_4"
	};
	const DataType bias1_4[CHAN_1]={
#include "bias1_4"
	};
	//BLOCK 5
	const DataType p0_5[CHAN_1][CHAN_1]={
#include "p0_5"
	};
	const DataType p1_5[CHAN_1][CHAN_1]={
#include "p1_5"
	};
	const DataType bias0_5[CHAN_1 * EXP]={
#include "bias0_5"
	};
	const DataType bias1_5[CHAN_1]={
#include "bias1_5"
	};
	//GROUP 3
	//BLOCK 6
	const DataType p0_6[CHAN_1][CHAN_2]={
#include "p0_6"
	};
	const DataType p1_6[CHAN_2][CHAN_2]={
#include "p1_6"
	};
	const DataType p2_6[CHAN_1][CHAN_2]={
#include "p2_6"
	};
	const DataType bias0_6[CHAN_2 * EXP]={
#include "bias0_6"
	};
	const DataType bias1_6[CHAN_2]={
#include "bias1_6"
	};
	const DataType bias2_6[CHAN_2]={
#include "bias2_6"
	};
	//BLOCK 7
	const DataType p0_7[CHAN_2][CHAN_2]={
#include "p0_7"
	};
	const DataType p1_7[CHAN_2][CHAN_2]={
#include "p1_7"
	};
	const DataType bias0_7[CHAN_2 * EXP]={
#include "bias0_7"
	};
	const DataType bias1_7[CHAN_2]={
#include "bias1_7"
	};
	//BLOCK 8
	const DataType p0_8[CHAN_2][CHAN_2]={
#include "p0_8"
	};
	const DataType p1_8[CHAN_2][CHAN_2]={
#include "p1_8"
	};
	const DataType bias0_8[CHAN_2 * EXP]={
#include "bias0_8"
	};
	const DataType bias1_8[CHAN_2]={
#include "bias1_8"
	};

	const DataType p_9[DIM_2 * DIM_2* CHAN_2][N]={
#include "p_9"
	};
	const DataType bias_9[N]={
#include "bias_9"
	};
}
