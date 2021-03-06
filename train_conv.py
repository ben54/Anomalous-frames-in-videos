from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import load_data as ld
from net import Network
from plots import plotLC

ALPHA = 1e-2
BATCH_SIZE = 20
NITER = 12000

# load test data
test = np.load('test.npy')
test_labels = ld.one_hot(np.load('test_labels.npy'))

# initialize network
n = Network(ALPHA)
# load training and validation datasets
D = ld.DataSet()

train_accuracies = []
batch_accuracies = []
valid_batch_accuracies = []
valid_accuracies = []

# train the network
for i in range(NITER):
	batch = D.next_batch(BATCH_SIZE)
	batch_accuracies.append(n.getAccuracy(batch[0], batch[1]))
	if i > 0 and i % (D.get_trainsize() / BATCH_SIZE) == 0:
		train_accuracies.append(sum(batch_accuracies) / len(batch_accuracies))
		for j in range((int)(D.get_validsize()/BATCH_SIZE)):
			valid_batch = D.next_valid_batch(BATCH_SIZE)
			valid_batch_accuracies.append(n.getAccuracy(valid_batch[0], valid_batch[1]))
		valid_accuracies.append(sum(valid_batch_accuracies) / len(valid_batch_accuracies))
		batch_accuracies = []
		valid_batch_accuracies = []
		print("Training accuracy %g" % (train_accuracies[-1]))
	n.step(batch[0], batch[1])

# plot learning curves
plotLC(train_accuracies, range(1, (int) (NITER / (D.get_trainsize() / BATCH_SIZE)) + 1), "TrainingEpochs")
plotLC(valid_accuracies, range(1, (int) (NITER / (D.get_trainsize() / BATCH_SIZE)) + 1), "ValidationEpochs")
