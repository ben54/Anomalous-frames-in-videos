# Detecting anomalous frames in a video

This project aims to explore using deep learning to detect anomalous frames in a video based on content of the frame, possibly neighboring frames. 

The dataset being used is the pedestrian dataset from UCSD which is a video of a pedestrian walkway from a fixed camera location. Anomalous frames are those which contain something other than pedestrains and the background, for example, frames with cars or bikes.

One of the challenges here is that we're hoping to determine behavior as anomalous based on the presence of something in the frame that's not non-anomalous. That is, instead of treating this as an object detection probelm where we could train car and bike recongnizers and just flag frames where these appear as anomalous, we're saying we want to even capture a frame with a new vehicle that one has never seen before (say, a tank) as anomalous.

Right now we're proceeding with the simple approach of treating each frame as an image and trying to do image classification with a CNN. Of course, we'll probably want to move on to an approach that eventually leverages the sequential nature of the video, maybe using an RNN.

Baseline: The percentage of non-anomalous frames is around 55.45% so a model that always predicts non-anomalous will achieve this as the accuracy.
