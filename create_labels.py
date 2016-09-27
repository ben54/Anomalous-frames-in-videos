import numpy as np

# ground truth of which frames are anomalous from Matlab file
labels = []
labels.append(range(59,146))
labels.append(range(57,162))
labels.append(range(80,201))
labels.append([])
labels.append(range(9,84) + range(139,201))
labels.append(range(0,101) + range(109,201))
labels.append(range(0,176))
labels.append(range(0,95))
labels.append(range(0,49))
labels.append(range(0,141))
labels.append(range(69,164))
labels.append(range(129,201))
labels.append(range(0,157))
labels.append(range(0,201))
labels.append(range(137,201))
labels.append(range(122,201))
labels.append(range(0,48))
labels.append(range(53,121))
labels.append(range(63,139))
labels.append(range(44,176))
labels.append(range(30,201))
labels.append(range(15,108))
labels.append(range(7,166))
labels.append(range(49,172))
labels.append(range(39,136))
labels.append(range(76,145))
labels.append(range(9,123))
labels.append(range(104,201))
labels.append(range(0,16) + range(44,114))
labels.append(range(174,201))
labels.append(range(0,181))
labels.append(range(0,53) + range(64,114))
labels.append(range(4,166))
labels.append(range(0,122))
labels.append(range(85,201))
labels.append(range(14,109))

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

train_labels = np.asarray(train_labels).reshape((len(train_labels), 1))
test_labels = np.asarray(test_labels).reshape((len(test_labels), 1))

np.save('train_labels.npy', train_labels)
np.save('test_labels.npy', test_labels)
