import numpy as np
import tensorflow as tf

def choice(size):
	dx = np.zeros(size, dtype=np.int)
	dy = np.zeros(size, dtype=np.int)
	k = np.zeros([3,3, size, size], dtype=np.int)
	choice = np.array([-2, -1,0,1, 2]);
	d = np.random.choice(choice, size=size)
	for i in range(size):
		if d[i] == -2:
			dx[i] = 0
			dy[i] = -1
			k[1][2][i][i] = 1
		elif d[i] == -1:
			dx[i] = -1
			dy[i] = 0
			k[2][1][i][i] = 1
		elif d[i] == 0:
			dx[i] = 0
			dy[i] = 0
			k[1][1][i][i] = 1
		elif d[i] == 1:
			dx[i] = 1
			dy[i] = 0
			k[0][1][i][i] = 1
		elif d[i] == 2:
			dx[i] = 0
			dy[i] = 1
			k[1][0][i][i] = 1
	return dx, dy, k

def shift(tensor, kx, ky):
	dim = tensor.shape
	out = np.zeros(dim)
	assert(tensor.ndim == 3)
	assert(ky.size == kx.size == dim[2])
	for i in range(dim[0]):
		for j in range(dim[1]):
			for k in range(dim[2]):
				di = i - kx[k]
				dj = j - ky[k]
				if di <0 or di >= dim[0] or dj <0 or dj >=dim[1]:
					continue
				out[i, j, k] = tensor[di, dj, k]
	return out

def linear_cb(tensor, p):
	dimT = tensor.shape
	dimP = p.shape
	assert(tensor.ndim == 3)
	assert(p.ndim == 2)
	assert(dimT[2] == dimP[0])
	act = np.zeros([dimT[0],dimT[1], dimP[1]])
	for i in range(dimT[0]):
		for j in range(dimT[1]):
			for n in range(dimP[1]):
				for k in range(dimT[2]):
					act[i, j, n] = act[i, j, n] + p[k, n] * tensor[i, j, k]
	return act

def Relu(tensor,ave,std):
	dim = tensor.shape
	dimA = ave.shape
	dimD = std.shape
	assert(tensor.ndim == 3)
	assert(ave.ndim == 1)
	assert(std.ndim == 1)
	assert(dim[2] == dimA[0] == dimD[0])
	re = np.zeros(dim)
	for i in range(dim[0]):
		for j in range(dim[1]):
			for n in range(dim[2]):
				norm = (tensor[i,j,n] - ave[n])/std[n]
				re[i,j,n] = max(norm, 0)
	return re

def main():
	D = 9
	M = 16
	N = 8

	Dx, Dy, k_shift = choice(M)
	np.savetxt("dx", Dx.reshape([1, -1]), delimiter=',', fmt='%d')
	np.savetxt("dy", Dy.reshape([1, -1]), delimiter=',', fmt='%d')


	tensor = np.random.rand(D, D, M)
	
	np.savetxt("input", tensor.reshape([1, -1]), delimiter=',')
	
	p = np.random.rand(M, N)
	np.savetxt("p", p.reshape([1, -1]), delimiter=',')


	gensor = shift(tensor, Dx, Dy)
	np.savetxt("g", gensor.reshape([1, -1]), delimiter=',')

	act = linear_cb(gensor, p)
	np.savetxt("fmap", act.reshape([1, -1]), delimiter=',')
	fmap = act.reshape([D*D, N])
	ave = np.mean(fmap, axis = 0)
	std = np.std(fmap, axis = 0);
	np.savetxt("ave", ave.reshape([1, -1]), delimiter=',')
	np.savetxt("std", std.reshape([1, -1]), delimiter=',')
	out = Relu(act, ave,std)
	np.savetxt("ref", out.reshape([1, -1]), delimiter=',')


	g =tf.Graph()
	with g.as_default():
		t_im = tf.constant(tensor, dtype=tf.float32, shape=[1,D,D,M])
		p_conv = tf.constant(p, dtype=tf.float32, shape=[1,1,M, N])
		kernel = tf.constant(k_shift, shape=[3,3,M, M])
		t_shift = tf.nn.conv2d(t_im, k_shift, strides=[1,1,1,1], padding="SAME")
		t_conv = tf.nn.conv2d(t_shift,p_conv,strides=[1,1,1,1], padding="SAME")
		t_pool = tf.nn.max_pool(t_conv,[1, 2, 2, 1],[1,2,2,1], "VALID")
		t_relu = tf.nn.relu(t_pool)
	with tf.Session(graph=g) as sess:
		t_gen = t_pool.eval()
		np.savetxt("pool", t_gen.reshape([1, -1]), delimiter=',')

if __name__ == "__main__":
	main()
