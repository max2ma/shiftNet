#include <iostream>
#include <fstream>
#include <cmath>
#include "hls_stream.h"
#include "para.h"
using namespace std;
using namespace para;

extern "C"
void shift(float *tensor, float* act);

int main(){
	float input[D *D*C] = {
#include "inputs_batch_0"
		//#include "t_im"
	};

	float out[N];
	float ref[N] = {
//#include "outputs_batch_0"
#include "t_l1"
		//#include "t_cifar"
	};

	int err = 0, TT =N;
	float ave = 0;

	shift(input, out);

	float eps = 1e-6;


	for(int k=0;k<N;k++){
		if(ref[k] <= eps){
			TT --;
			if(out[k] <=eps)
				continue;
			else{
				err ++;
		cout	<<k<<','
			<<out[k]<<','
			<<ref[k] << ','
			<<endl;
				continue;
			}
		}
		float diff = abs(out[k] / ref[k]- 1);// ref[i][j][k]);
		ave+=diff;
		if (diff > 1e-1){
			err ++;
		cout	<<k<<','
			<<out[k]<<','
			<<ref[k] << ','
			<<endl;
		}
	}
	cout << "there are in total " << err << " errors."<<endl;
	cout << "the ave error is " << ave/TT << " ."<<endl;
	if(err == 0){
		return 0;
	}
	else {
		return -1;
	}
}
