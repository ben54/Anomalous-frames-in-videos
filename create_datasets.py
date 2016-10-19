from PIL import Image
import os
import re
import numpy as np

data_dir = "/Users/tnybny/Documents/Anomaly detection in video/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test"
fldrs = os.listdir(data_dir)
imlist = []
for fdname in fldrs:
    p = re.compile("^Test0\d\d$")
    if p.match(fdname):
        fnames = os.listdir(data_dir + "/" + fdname)
	for f in fnames:
		q = re.compile("^\d\d\d.tif$")
    		if q.match(f):
			im = np.array(Image.open(data_dir + "/" + fdname + "/" + f), dtype = "float32")
			im = np.divide(np.subtract(im, 128), 128)
			imlist.append(im)
			# add mirrored copy of image
			imlist.append(np.fliplr(im))

# 2/3rd 1/3rd split set of images available into 
# train (contains training + validation actually) and test
train = imlist[:24*200*2]
test = imlist[24*200*2:]

train = np.asarray(train, dtype = "float32")
train = train.reshape(train.shape[0], train.shape[1] * train.shape[2])
test = np.asarray(test, dtype = "float32")
test = test.reshape(test.shape[0], test.shape[1] * test.shape[2])

np.save('train.npy', train)
np.save('test.npy', test)
