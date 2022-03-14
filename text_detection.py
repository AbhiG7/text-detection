import numpy as np
import cv2
import tensorflow as tf



train_data = np.loadtxt('C:\mnist_dataset\mnist_train.csv', delimiter=",")
#test_data = np.loadtxt('C:\mnist_dataset\mnist_test.csv', delimiter=",")

train_imgs = train_data[:, 1:].astype(np.uint8)
train_labels = train_data[:, :1].astype(np.uint8)
#test_labels = test_data[:, :1].astype(np.uint8)
#test_imgs = test_data[:, 1:].astype(np.uint8)

winSize = (28,28)
blockSize = (10,10)
blockStride = (6,6)
cellSize = (10,10)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True
 
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)

for i in range(10):
    if train_labels[i, 0] == 0:
        descriptor_0 = hog.compute(train_imgs[i, :])
    if train_labels[i, 0] == 1:
        descriptor_1 = hog.compute(train_imgs[i, :])
    if train_labels[i, 0] == 2:
        descriptor_2 = hog.compute(train_imgs[i, :])
    if train_labels[i, 0] == 3:
        descriptor_3 = hog.compute(train_imgs[i, :])
    if train_labels[i, 0] == 4:
        descriptor_4 = hog.compute(train_imgs[i, :])
    if train_labels[i, 0] == 5:
        descriptor_5 = hog.compute(train_imgs[i, :])
    if train_labels[i, 0] == 6:
        descriptor_6 = hog.compute(train_imgs[i, :])
    if train_labels[i, 0] == 7:
        descriptor_7 = hog.compute(train_imgs[i, :])
    if train_labels[i, 0] == 8:
        descriptor_8 = hog.compute(train_imgs[i, :])
    if train_labels[i, 0] == 9:
        descriptor_9 = hog.compute(train_imgs[i, :])


