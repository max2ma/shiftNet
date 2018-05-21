#pragma once

// M2S
template<int M, int C, int REP, typename T_IN, typename T_OUT>
void M2S(T_IN *mem, hls::stream<T_OUT> s_mem[C]){
#pragma HLS INLINE
	for(int rep = 0; rep < REP; rep++)
		for(int i=0;i<M;i++)
				for(int c=0;c<C;c++){
#pragma HLS pipeline
					s_mem[c].write((T_OUT)mem[rep*M*C + i*C+c]);
				}
}



// S2M
template<int M, int C, int REP, typename T_IN, typename T_OUT>
void S2M(hls::stream<T_IN> s_mem[C], T_OUT *mem){
#pragma HLS INLINE
	for(int rep = 0; rep < REP; rep++)
		for(int i=0;i<M;i++)
				for(int c=0;c<C;c++)
#pragma HLS pipeline
					mem[rep * M *C + i*C + c]= (T_OUT)s_mem[c].read();
}
