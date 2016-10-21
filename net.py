import tensorflow as tf

# number of feature maps at input and each of the layers in progression
NFEATS = 1
L1 = 64
L2 = 128
L3 = 128
L4 = 128
NLABELS = 2
IMG_WIDTH = 158
IMG_HEIGHT = 238

# w.r.t dropout, probability of retaining a node
KEEP_PROB = 0.5

def weight_variable(shape):
	# create weight tensor (kernel) given 4-D shape of 
	# kernel dim1, kernel dim2, fan_in and fan_out
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	# create bias variable given 1-D shape of fan_out
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W):
	# convolve a weight tensor over an input image
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
	# perform 2x2 max pooling on a given tensor
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

class Network:
	def __init__(self, alpha):
		self.x = tf.placeholder(tf.float32, [None, IMG_WIDTH * IMG_HEIGHT])
		self.y_ = tf.placeholder(tf.float32, [None, NLABELS])
		self.keep_prob = tf.placeholder(tf.float32)
    
		self.x_image = tf.reshape(self.x, [-1, IMG_WIDTH, IMG_HEIGHT, 1])

		# first conv layer
		self.W_conv1 = weight_variable([5, 5, NFEATS, L1])
		self.b_conv1 = bias_variable([L1])

		self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
		
		# pooling
		self.h_pool1 = max_pool_2x2(self.h_conv1)

		# second conv layer
		self.W_conv2 = weight_variable([5, 5, L1, L2])
		self.b_conv2 = bias_variable([L2])

		self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
		
		# pooling
		self.h_pool2 = max_pool_2x2(self.h_conv2)

		# third conv layer
		self.W_conv3 = weight_variable([5, 5, L2, L3])
		self.b_conv3 = bias_variable([L3])

		self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self.W_conv3) + self.b_conv3)
		
		# pooling
		self.h_pool3 = max_pool_2x2(self.h_conv3)
		
		# fully connected relu layer
		self.W_fc1 = weight_variable([20 * 30 * L3, L4])
		self.b_fc1 = bias_variable([L3])
		
		self.h_pool3_flat = tf.reshape(self.h_pool3, [-1, 20 * 30 * L3])
		
		self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool3_flat, self.W_fc1) + self.b_fc1)

		# dropout
		self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

		# softmax
		self.W_fc2 = weight_variable([L4, NLABELS])
		self.b_fc2 = bias_variable([NLABELS])

		self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)
		
		# list of trainable parameters
		self.trainable = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2,
				self.W_conv3, self.b_conv3,
				self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]

		# count when prediction = actual
		self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))	
	
		# avg of negative logloss
		self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices = [1]))
		self.train_step = tf.train.AdamOptimizer(alpha).minimize(self.cross_entropy)
	
		self.sess = tf.Session()	
		self.sess.run(tf.initialize_all_variables())

	def getAccuracy(self, images, labels):
		# evaluate the accuracy for a given batch of pairs of images and labels
		# with the current set of weights
		return self.accuracy.eval(feed_dict = {self.x: images, self.y_: labels, self.keep_prob: 1.0}, session = self.sess)

	def step(self, images, labels):
		# perform a step of forward prop followed by backprop for the batch
		# of pairs of images and labels given
		self.train_step.run(feed_dict = {self.x: images, self.y_: labels, self.keep_prob: KEEP_PROB}, session = self.sess)

	def setParams(self, params):
		# load the given weights into the neural network
   		self.sess.run([tf.assign(v, x) for v, x in zip(self.trainable, params)])
