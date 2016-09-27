from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
#import cPickle

import load_data as ld
from net import Network
from plots import plotLC

ALPHA = 1e-3
BATCH_SIZE = 200
NITER = 5000

test = np.load('test.npy')
test_labels = np.load('test_labels.npy')

n = Network(ALPHA)
D = ld.DataSet()

train_accuracies = []
batch_accuracies = []

for i in range(NITER):
	batch = D.next_batch(BATCH_SIZE)
	batch_accuracies.append(n.getAccuracy(batch[0], batch[1]))
	if i > 0 and i % 48 == 0:
		train_accuracies.append(sum(batch_accuracies)/len(batch_accuracies))
		batch_accuracies = []
		print("Training accuracy %g" % (train_accuracies[-1]))
	n.step(batch[0], batch[1])

plotLC(train_accuracies, range(1, NITER + 1), "TrainingMinibatches")

#print("test accuracy %g" % accuracy.eval(feed_dict={
#    x: test, y_: test_labels, keep_prob: 1.0}))
