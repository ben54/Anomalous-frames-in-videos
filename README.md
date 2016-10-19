# Detecting anomalous frames in a video

This project aims to explore using deep learning to detect anomalous frames in a video based on content of the frame, possibly neighboring frames. 

The dataset being used is the pedestrian dataset from UCSD which is a video of a pedestrian walkway from a fixed camera location. Anomalous frames are those which contain something other than pedestrains and the background, for example, frames with cars or bikes.

One of the challenges here is that we're hoping to determine behavior as anomalous based on the presence of something in the frame that's not non-anomalous. That is, instead of treating this as an object detection probelm where we could train car and bike recongnizers and just flag frames where these appear as anomalous, we're saying we want to even capture a frame with a new vehicle that one has never seen before (say, a tank) as anomalous.

Right now we're proceeding with the simple approach of treating each frame as an image and trying to do image classification with a CNN. Of course, we'll probably want to move on to an approach that eventually leverages the sequential nature of the video, maybe using an RNN.

Baseline: The percentage of non-anomalous frames is around 55.45% so a model that always predicts non-anomalous will achieve this as the accuracy.

## Description of each .py file in this repository

create\_datasets.py reads the raw images from tif files, creates mirrored copies of each image and writes all the images back as numpy arrays into train.npy and test.npy for fast reading in the future.

create\_labels.py takes hard coded labels for images (1 for anomalous, 0 for non-anomalous), accounts for the mirroring that happens in create\_datasets.py and writes as numpy arrays into train\_labels.npy and test\_labels.npy for fast reading in the future.

net.py contains the class that holds the CNN along with the complete architecture specification, number of feature maps, loss function, weight initializations and optimization method. The class also contains utility functions to compute accuracy for a given batch, perform a step of fwd prop followed by backprop for a given batch and load state of weight tensor provided.

load\_data.py contains the class that encapsulates functions relevant to loading training and validation data, retrieving batches of pairs of images and labels, and utility functions to access some properties of these datasets.

train\_conv.py is the 'main' function. It defines runtime parameters such as learning rate, batch size and number of epochs, creates objects of network and data, runs the network for those epochs, records training and validation accuracies and then plots learning curves.

plots.py contains functions that plot learning curves.
