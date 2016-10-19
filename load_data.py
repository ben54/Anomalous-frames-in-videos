import numpy as np

class DataSet(object):
	def __init__(self):
		self._index_in_epoch = 0
		self._index_valid = 0
		self._epochs_completed = 0
		self.train = np.load('train.npy')
		self.train_labels = np.load('train_labels.npy')
		# take different random 80/20 train/valid split for every class object
      		perm = np.arange(self.train.shape[0])
      		np.random.shuffle(perm)
      		self.train = self.train[perm]
      		self.train_labels = self.train_labels[perm]
		self.valid = self.train[(int)(0.8 * self.train.shape[0]):, :]
		self.train = self.train[:(int)(0.8 * self.train.shape[0]), :]
		self.valid_labels = self.train_labels[(int)(0.8 * self.train_labels.shape[0]):]
		self.train_labels = self.train_labels[:(int)(0.8 * self.train_labels.shape[0])]
		self._num_examples = self.train.shape[0]

	def next_batch(self, batch_size):
		# for training data, given batch size, return a batch of pairs of images and labels
		start = self._index_in_epoch
    		self._index_in_epoch += batch_size
    		if self._index_in_epoch > self._num_examples:
      			# Finished epoch
     			self._epochs_completed += 1
      			# Shuffle the data
      			perm = np.arange(self._num_examples)
      			np.random.shuffle(perm)
      			self.train = self.train[perm]
      			self.train_labels = self.train_labels[perm]
      			# Start next epoch
      			start = 0
      			self._index_in_epoch = batch_size
      			assert batch_size <= self._num_examples
    		end = self._index_in_epoch
    		return self.train[start:end], self.train_labels[start:end]
	
	def next_valid_batch(self, batch_size):
		# for validation data, given batch size, return a batch of pairs of images and labels
		start = self._index_valid
		self._index_valid += batch_size
		end = self._index_valid
		return self.valid[start:end], self.valid_labels[start:end]

	def get_trainsize(self):
		# number of training images
		return self.train.shape[0]
	
	def get_validsize(self):
		# number of validation images
		return self.valid.shape[0]
