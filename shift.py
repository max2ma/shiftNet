import numpy as np
import tensorflow as tf

def choice(size):
	dx = np.zeros(size, dtype=np.int)
	dy = np.zeros(size, dtype=np.int)
	k = np.zeros([3,3, size, size], dtype=np.int)
	choice = np.array([-2, -1,0,1, 2], dtype=np.int);
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
	return d, k

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
	D = 11
	C = 3
	M = 8
	N = 16

	Dx, k_shift = choice(M)
	np.savetxt("d", Dx.reshape([1, -1]), delimiter=',', fmt='%d')


	tensor = np.random.rand(D, D, C)
	
	np.savetxt("input", tensor.reshape([1, -1]), delimiter=',')
	
	p0 = np.random.rand(C, M)
	np.savetxt("p0", p0.reshape([1, -1]), delimiter=',')
	
	p1 = np.random.rand( M, N)
	np.savetxt("p1", p1.reshape([1, -1]), delimiter=',')


	def block(idx ,fmap, in_c, out_c, ex,  stride):
		p0 = np.random.rand(1, 1, in_c, in_c * ex)
		np.savetxt("p0_%d"%idx, p0.reshape([1, -1]), delimiter=',')
		t_conv0 = tf.nn.conv2d(fmap,p0,strides=[1,1,1,1],
					padding="VALID")
		t_relu=tf.nn.relu(t_conv0)
		t_shift=shift(idx, t_relu, in_c*ex)
		p1 = np.random.rand(1, 1, in_c * ex, out_c)
		np.savetxt("p1_%d"%idx, p1.reshape([1, -1]), delimiter=',')
		t_act = tf.nn.relu(tf.nn.conv2d(t_shift,p1,strides=[1,stride, stride,1],
					padding="VALID"), name="block_%d"%idx)
		return t_act


	def shift(idx, t_im, in_c):

		d, k_shift = choice(in_c)
		np.savetxt("d_%d"%idx, d.reshape([1, -1]), delimiter=',', fmt='%d')
		t_shift = tf.nn.conv2d(t_im, k_shift, strides=[1,1,1,1], padding="SAME")
		return t_shift
	g =tf.Graph()
	with g.as_default():
		t_im = tf.constant(tensor, dtype=tf.float32, shape=[1,D,D,C])
		t_act = block(0, t_im, C, N, 2, 2)
		#t_act=shift(0,t_im, C)
	with tf.Session(graph=g) as sess:
		t_gen= sess.run(t_act)
		np.savetxt("t_act", t_gen.reshape([1, -1]), delimiter=',')

if __name__ == "__main__":
	main()
