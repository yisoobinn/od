import tensorflow as tf 
from tensorflow.keras import *
import numpy as np 
import cv2
import os
import selectivesearch

x_train_color = np.zeros((80, 160, 160, 3), np.uint8)
x_train = np.zeros((80, 160, 160), np.uint8)

y_train = np.zeros(80, np.uint8)

input_shape = (160, 160, 1)

i = 0
face_filelist = os.listdir("./dataset/face")
for face_file in face_filelist:
    x_train_color[i] = cv2.imread("./dataset/face/{}".format(face_file)) # (height, width, channels) = (160, 160, 3)
    i = i + 1

wo_face_filelist = os.listdir("./dataset/wo_face")
for wo_face_file in wo_face_filelist:
    x_train_color[i] = cv2.imread("./dataset/wo_face/{}".format(wo_face_file)) # (height, width, channels) = (160, 160, 3)
    i = i + 1

for i in range(0, 80):
    x_train[i] = cv2.cvtColor(x_train_color[i], cv2.COLOR_BGR2GRAY)

print(x_train.shape) 

for i in range(0, 40):
    y_train[i] = 0

for i in range(40, 80):
    y_train[i] = 1

# print(y_train.shape) (80,)
model = tf.keras.models.Sequential([
    Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=100)
model.save("seoul.h5")

