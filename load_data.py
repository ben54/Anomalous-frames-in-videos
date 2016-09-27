import numpy as np

class DataSet(object):
	def __init__(self):
		self._index_in_epoch = 0
		self._epochs_completed = 0
		self.train = np.load('train.npy')
		self.train_labels = np.load('train_labels.npy')
		self._num_examples = self.train.shape[0]

	def next_batch(self, batch_size):
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
