import tensorflow as tf

NFEATS = 1
L1 = 16
L2 = 32
L3 = 64
KEEP_PROB = 0.5

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

class Network:
	def __init__(self, alpha):
		self.x = tf.placeholder(tf.float32, [None, 37604])
		self.y_ = tf.placeholder(tf.float32, [None, 1])
		self.keep_prob = tf.placeholder(tf.float32)
    
		self.x_image = tf.reshape(self.x, [-1, 158, 238, 1])

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

		# fully connected relu layer
		self.W_fc1 = weight_variable([40 * 60 * L2, L3])
		self.b_fc1 = bias_variable([L3])
		
		self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 40 * 60 * L2])
		
		self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

		# dropout
		self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

		# softmax
		self.W_fc2 = weight_variable([L3, 1])
		self.b_fc2 = bias_variable([1])

		self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)
		
		# list of trainable parameters
		self.trainable = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2,
				self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]

		# count when prediction = actual
		self.correct_prediction = tf.equal(self.y_conv, self.y_)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))	
	
		# Avg of negative logloss
		self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices = [1]))
		self.train_step = tf.train.AdamOptimizer(alpha).minimize(self.cross_entropy)
		
		self.config = tf.ConfigProto(device_count = {'GPU': 0}) 
		self.sess = tf.Session(config = self.config)
		self.sess.run(tf.initialize_all_variables())

	def getAccuracy(self, images, labels):
		return self.accuracy.eval(feed_dict = {self.x: images, self.y_: labels, self.keep_prob: 1.0}, session = self.sess)

	def step(self, images, labels):
		self.train_step.run(feed_dict = {self.x: images, self.y_: labels, self.keep_prob: KEEP_PROB}, session = self.sess)

	def setParams(self, params):
   		 self.sess.run([tf.assign(v, x) for v, x in zip(self.trainable, params)])
