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
	D = 32
	C = 3
	tensor = np.random.normal(size=(D, D, C))
	np.savetxt("input", tensor.reshape([1, -1]), delimiter=',')

	fc = np.random.normal(size=(8, 8, 64, 10))
	np.savetxt("p_9", fc.reshape([1, -1]), delimiter=',')
	
	def block(idx ,fmap, in_c, out_c, ex,  stride):
		p0 = np.random.normal(scale=1/np.sqrt(in_c),size=(1, 1, in_c, in_c * ex))
		np.savetxt("p0_%d"%idx, p0.reshape([1, -1]), delimiter=',')
		t_conv0 = tf.nn.conv2d(fmap,p0,strides=[1,1,1,1],
					padding="VALID")
		t_relu=tf.nn.relu(t_conv0)
		t_shift=shift(idx, t_relu, in_c*ex)
		p1 = np.random.normal(scale=1/np.sqrt(out_c),size=(1, 1, in_c * ex, out_c))
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
		p_9 = tf.constant(np.reshape(fc, [-1, 10]), dtype=tf.float32)
		t_act0 = block(0, t_im, C, 16, 1, 1)
		t_act1 = block(1, t_act0, 16, 16, 1, 1)
		t_act2 = block(2, t_act1, 16, 16, 1, 1)
		t_act3 = block(3, t_act2, 16, 32, 1, 2)
		t_act4 = block(4, t_act3, 32, 32, 1, 1)
		t_act5 = block(5, t_act4, 32, 32, 1, 1)
		t_act6 = block(6, t_act5, 32, 64, 1, 2)
		t_act7 = block(7, t_act6, 64, 64, 1, 1)
		t_act8 = block(8, t_act7, 64, 64, 1, 1)
		t_arr = tf.reshape(t_act8, [1, -1])
		t_cifar = tf.nn.relu(tf.matmul(t_arr, p_9))
		#t_act=shift(0,t_im, C)
	with tf.Session(graph=g) as sess:
		n_cifar= sess.run(t_cifar)
		np.savetxt("t_cifar", n_cifar.reshape([1, -1]), delimiter=',')

if __name__ == "__main__":
	main()
