#pragma once
#include "para.h"
#include "ap_fixed.h"
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
	
	typedef ap_fixed<8, 0, AP_RND, AP_SAT> T_P0; // for [-0.5, 0.5)
	typedef ap_fixed<8, 1, AP_RND, AP_SAT> T_P1; // for [-1, 1)
	typedef ap_fixed<8, 2, AP_RND, AP_SAT> T_P2; // for[-2, 2)
	typedef ap_fixed<8, 3, AP_RND, AP_SAT> T_P3; // for [-4, 4)
	typedef ap_fixed<8, 4, AP_RND, AP_SAT> T_P4; // for [-8, 8)
	typedef ap_fixed<8, 5, AP_RND, AP_SAT> T_P5; // for [-16, 16)
	
	const T_P0 kernel[3][3][C][CHAN_0] = {
#include "t_k"
	};
	const T_P1 bias0[CHAN_0]={
#include "bias0"
	};
	//BLOCK 0
	const T_P1 p0_0[CHAN_0][CHAN_0 * EXP]={
#include "p0_0"
	};
	const T_P2 p1_0[CHAN_0 * EXP][CHAN_0]={
#include "p1_0"
	};
	const T_P2 bias0_0[CHAN_0 * EXP]={
#include "bias0_0"
	};
	const T_P2 bias1_0[CHAN_0]={
#include "bias1_0"
	};

	//BLOCK 1
	const T_P1 p0_1[CHAN_0][CHAN_0 * EXP]={
#include "p0_1"
	};
	const T_P2 p1_1[CHAN_0 * EXP][CHAN_0]={
#include "p1_1"
	};
	const T_P1 bias0_1[CHAN_0 * EXP]={
#include "bias0_1"
	};
	const T_P1 bias1_1[CHAN_0]={
#include "bias1_1"
	};

	//BLOCK 2
	const T_P1 p0_2[CHAN_0][CHAN_0]={
#include "p0_2"
	};
	const T_P2 p1_2[CHAN_0][CHAN_0]={
#include "p1_2"
	};
	const T_P1 bias0_2[CHAN_0 * EXP]={
#include "bias0_2"
	};
	const T_P1 bias1_2[CHAN_0]={
#include "bias1_2"
	};

	// GROUP 2
	//BLOCK 3
	const T_P2 p0_3[CHAN_0][CHAN_1]={
#include "p0_3"
	};
	const T_P2 p1_3[CHAN_1][CHAN_1]={
#include "p1_3"
	};
	const T_P2 p2_3[CHAN_0][CHAN_1]={
#include "p2_3"
	};
	const T_P2 bias0_3[CHAN_1 * EXP]={
#include "bias0_3"
	};
	const T_P1 bias1_3[CHAN_1]={
#include "bias1_3"
	};
	const T_P1 bias2_3[CHAN_1]={
#include "bias2_3"
	};
	//BLOCK 4
	const T_P1 p0_4[CHAN_1][CHAN_1]={
#include "p0_4"
	};
	const T_P2 p1_4[CHAN_1][CHAN_1]={
#include "p1_4"
	};
	const T_P1 bias0_4[CHAN_1 * EXP]={
#include "bias0_4"
	};
	const T_P1 bias1_4[CHAN_1]={
#include "bias1_4"
	};
	//BLOCK 5
	const T_P0 p0_5[CHAN_1][CHAN_1]={
#include "p0_5"
	};
	const T_P2 p1_5[CHAN_1][CHAN_1]={
#include "p1_5"
	};
	const T_P1 bias0_5[CHAN_1 * EXP]={
#include "bias0_5"
	};
	const T_P1 bias1_5[CHAN_1]={
#include "bias1_5"
	};
	//GROUP 3
	//BLOCK 6
	const T_P1 p0_6[CHAN_1][CHAN_2]={
#include "p0_6"
	};
	const T_P1 p1_6[CHAN_2][CHAN_2]={
#include "p1_6"
	};

	const T_P0 p2_6[CHAN_1][CHAN_2]={
#include "p2_6"
	};
	const T_P1 bias0_6[CHAN_2 * EXP]={
#include "bias0_6"
	};
	const T_P2 bias1_6[CHAN_2]={
#include "bias1_6"
	};
	const T_P1 bias2_6[CHAN_2]={
#include "bias2_6"
	};
	//BLOCK 7
	const T_P1 p0_7[CHAN_2][CHAN_2]={
#include "p0_7"
	};
	const T_P2 p1_7[CHAN_2][CHAN_2]={
#include "p1_7"
	};
	const T_P0 bias0_7[CHAN_2 * EXP]={
#include "bias0_7"
	};
	const T_P2 bias1_7[CHAN_2]={
#include "bias1_7"
	};
	//BLOCK 8
	const T_P1 p0_8[CHAN_2][CHAN_2]={
#include "p0_8"
	};

	const T_P3 p1_8[CHAN_2][CHAN_2]={
#include "p1_8"
	};
	const T_P1 bias0_8[CHAN_2 * EXP]={
#include "bias0_8"
	};
	const T_P1 bias1_8[CHAN_2]={
#include "bias1_8"
	};

	const T_P2 p_9[DIM_2 * DIM_2* CHAN_2][N]={
#include "p_9"
	};
	const T_P2 bias_9[N]={
#include "bias_9"
	};
}

