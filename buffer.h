#pragma once

#include "hls_video.h"

template<typename T,int H, int W, int C>
class ShiftBuffer{
	public:
		hls::Window<H, W, T> pixel[C];
		int ih[C], iw[C];

		ShiftBuffer(){
			for(int k = 0;k < C; k++){
				for(int i = 0;i < H; i++)
					for(int j = 0;j < W; j++)
#pragma HLS PIPELINE
							pixel[k].insert_pixel(0,i,j);
				ih[k] =0;
				iw[k] =0;
			}
		}

		const T insert(const T t, int c){
			int i = ih[c], j=iw[c];
			T tmp = pixel[c].getval(i,j);
			pixel[c].insert_pixel(t,i,j);
			j ++;
			if(j == W){
				j = 0;
				i ++;
				if(i == H)
					i=0;
			}
			ih[c] = i;
			iw[c] = j;
			return tmp;
		}
};
