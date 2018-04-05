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


def main():
	D = 9
	C = 3
	M = 8
	N = 16

	Dx, Dy, k_shift = choice(M)
	np.savetxt("dx", Dx.reshape([1, -1]), delimiter=',', fmt='%d')
	np.savetxt("dy", Dy.reshape([1, -1]), delimiter=',', fmt='%d')


	tensor = np.random.rand(D, D, C)
	
	np.savetxt("input", tensor.reshape([1, -1]), delimiter=',')
	
	p0 = np.random.rand(C, M)
	np.savetxt("p0", p0.reshape([1, -1]), delimiter=',')
	
	p1 = np.random.rand( M, N)
	np.savetxt("p1", p1.reshape([1, -1]), delimiter=',')


	g =tf.Graph()
	with g.as_default():
		t_im = tf.constant(tensor, dtype=tf.float32, shape=[1,D,D,C])
		p0_conv = tf.constant(p0, dtype=tf.float32, shape=[1,1,C, M])
		p1_conv = tf.constant(p1, dtype=tf.float32, shape=[1,1, M, N])
		kernel = tf.constant(k_shift, shape=[3,3,M, M])
		t_conv0 = tf.nn.conv2d(t_im,p0_conv,strides=[1,1,1,1], padding="SAME")
		t_shift = tf.nn.conv2d(t_conv0, k_shift, strides=[1,1,1,1], padding="SAME")
		t_conv1 = tf.nn.conv2d(t_shift,p1_conv,strides=[1,1,1,1], padding="SAME")
		t_pool = tf.nn.max_pool(t_conv1,[1, 2, 2, 1],[1,2,2,1], "VALID")
		t_relu = tf.nn.relu(t_pool)
	with tf.Session(graph=g) as sess:
		t_gen,t_po, t_map = sess.run([t_relu, t_pool, t_conv1])
		np.savetxt("relu", t_gen.reshape([1, -1]), delimiter=',')
		np.savetxt("fmap", t_map.reshape([1, -1]), delimiter=',')
		np.savetxt("pool", t_po.reshape([1, -1]), delimiter=',')

if __name__ == "__main__":
	main()
