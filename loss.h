#pragma once
#include <cmath>
template<int S>
float crossEntropyLoss(float sample[S], float target[S]){
	float loss = 0;
	for(int i = 0;i<S;i++)
		loss+= log(sample[i]) * target[i];
	return loss;
}


template<int N>
int ord_max(float sample[N]){
	int ord = 0;
	float tmp = sample[ord];
	for(int i = 1; i <N;i ++){
		float value = sample[i];
		if(value > tmp){
			tmp = value;
			ord = i;
		}
	}
	return ord;
}
