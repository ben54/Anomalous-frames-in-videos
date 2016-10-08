import numpy as np

# ground truth of which frames are anomalous from Matlab file
labels = []
labels.append(range(59,146))
labels.append(range(57,162))
labels.append(range(81,200))
labels.append([])
labels.append(range(9,84) + range(102,200))
labels.append(range(0,91) + range(111,200))
labels.append(range(0,171))
labels.append([])
labels.append([])
labels.append(range(0,141))
labels.append([])
labels.append([])
labels.append(range(0,141))
labels.append(range(0,200))
labels.append(range(139,200))
labels.append(range(129,200))
labels.append(range(0,44))
labels.append([])
labels.append(range(54,136))
labels.append(range(46,171))
labels.append(range(45,200))
labels.append([])
labels.append(range(2,162))
labels.append(range(44,168))
labels.append([])
labels.append(range(8,200))
labels.append(range(9,118))
labels.append(range(103,200))
labels.append(range(0,7) + range(43,111))
labels.append(range(159,200))
labels.append(range(0,141))
labels.append(range(0,50) + range(54,113))
labels.append(range(3,180))
labels.append([])
labels.append([])
labels.append(range(14,106))

# 2/3rd 1/3rd split
train_labels = [0] * 24 * 200
test_labels = [0] * 12 * 200

for imnum in range(0, len(labels)):
	if imnum < 24:
		for i in labels[imnum]:
			train_labels[imnum * 200 + i] = 1
	else:
		for i in labels[imnum]:
			test_labels[(imnum - 24) * 200 + i] = 1

# duplicate each member to acount for the mirroring that was done in create_datasets.py
train_labels = [val for val in train_labels for _ in (0, 1)]
test_labels = [val for val in test_labels for _ in (0, 1)]

train_labels = np.asarray(train_labels).reshape((len(train_labels), 1))
test_labels = np.asarray(test_labels).reshape((len(test_labels), 1))

np.save('train_labels.npy', train_labels)
np.save('test_labels.npy', test_labels)
