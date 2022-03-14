import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Load model that was previously created
model = tf.keras.models.load_model('digits_detect.model')

# Runs model prediction for each image file in the directory
for num in os.listdir():
    if ".png" in num:
        # Uses OpenCV and numpy to read image and invert it to make it black on white
        img = cv2.imread(num)[:,:,0]
        img = np.invert(np.array([img]))
        # Runs model prediction and prints output
        prediction = model.predict(img)
        # print(f'You probably gave me a {np.argmax(prediction)}')
        plt.text(0.5,-2,f'Computer says: You probably gave me a {np.argmax(prediction)}',size=12)
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()