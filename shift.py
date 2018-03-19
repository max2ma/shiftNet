import numpy as np

def main():
	D = 8
	M = 2
	N = 4

	Dx = np.random.randint(low=-D, high=D, size=M)
	np.savetxt("dx", Dx.reshape([1, -1]), delimiter=',', fmt='%d')
	Dy = np.random.randint(low=-D, high=D, size=M)
	np.savetxt("dy", Dy.reshape([1, -1]), delimiter=',', fmt='%d')
	tensor = np.random.rand(D, D, M)
	
	np.savetxt("input", tensor.reshape([1, -1]), delimiter=',')
	
	p = np.random.rand(M, N)
	np.savetxt("p", p.reshape([1, -1]), delimiter=',')

	gensor = np.zeros((D, D, M))
	act = np.zeros((D, D, N))

	for i in range(D):
		for j in range(D):
			for k in range(M):
				di = i - Dx[k]
				dj = j - Dy[k]
				if di <0 or di >= D or dj <0 or dj >=D:
					continue
				gensor[i, j, k] = tensor[di, dj, k]
	np.savetxt("g", gensor.reshape([1, -1]), delimiter=',')

	for i in range(D):
		for j in range(D):
			for n in range(N):
				for k in range(M):
					act[i, j, n] = act[i, j, n] + p[k, n] * gensor[i, j, k]
	
	np.savetxt("fmap", act.reshape([1, -1]), delimiter=',')
	fmap = act.reshape([D*D, N])
	ave = np.mean(fmap, axis = 0)
	std = np.std(fmap, axis = 0);
	np.savetxt("ave", ave.reshape([1, -1]), delimiter=',')
	np.savetxt("std", std.reshape([1, -1]), delimiter=',')
	for i in range(D):
		for j in range(D):
			for n in range(N):
				norm = (act[i,j,n] - ave[n])/std[n]
				act[i,j,n] = max(norm, 0)
	np.savetxt("ref", act.reshape([1, -1]), delimiter=',')

if __name__ == "__main__":
	main()
