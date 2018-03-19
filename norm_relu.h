#pragma once

template<typename T, int D, int C>
void _norm_relu(T fmap[D][D][C], const T ave[C], const T std[C], T act[D][D][C]){
	for(int i=0;i<D;i++){
		for(int j=0;j<D;j++){
			for(int k=0;k<C;k++){
#pragma HLS PIPELINE
				T diff = fmap[i][j][k] - ave[k];
				if(diff < 0)
					act[i][j][k] = 0;
				else
					act[i][j][k] = diff/std[k];
			}
		}
	}
}
