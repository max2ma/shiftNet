#include <iostream>
#include <cmath>
#include "para.h"
using namespace std;
using namespace para;


extern
void shift(DataType tensor[D][D][M], DataType act[D][D][N]);

int main(){
	DataType input[D][D][M] = {
#include "input"
	};
	float ref[D][D][N] = {
#include "ref"
	};


	DataType output[D][D][N];


	shift(input, output);

	int err = 0;
	float ave = 0;
	for(int i=0;i<D;i++)
		for(int j=0;j<D;j++)
			for(int k=0;k<N;k++){
#ifdef FIXED
				float diff = abs(output[i][j][k].to_float() - ref[i][j][k]);
#else
				float diff = abs(output[i][j][k] - ref[i][j][k]);
#endif
				ave+=diff;
				if (diff > 1e-2)
					err ++;
			}
	cout << "there are in total " << err << " errors."<<endl;
	cout << "the ave error is " << ave/D/D/N << " ."<<endl;

	if(err == 0)
		return 0;
	else
		return -1;
}
