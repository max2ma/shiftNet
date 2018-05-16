#pragma once

// M2S
template<typename T_IN, typename T_OUT, int DIM, int C>
void M2S(T_IN *mem, hls::stream<T_OUT> s_mem[C]){
#pragma HLS INLINE
	for(int i=0;i<DIM;i++)
		for(int j=0;j<DIM;j++)
			for(int c=0;c<C;c++){
#pragma HLS pipeline
				s_mem[c].write((T_OUT)mem[i*DIM*C+j*C+c]);
			}
}



// S2M
template<typename T_IN, typename T_OUT, int DIM, int C>
void S2M(hls::stream<T_IN> s_mem[C], T_OUT *mem){
#pragma HLS INLINE
	for(int i=0;i<DIM;i++)
		for(int j=0;j<DIM;j++)
			for(int c=0;c<C;c++)
#pragma HLS pipeline
				mem[i*DIM*C+j*C+c]= (T_OUT)s_mem[c].read();
}



