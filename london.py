import tensorflow as tf
import numpy as np 
import cv2 


model = tf.keras.models.load_model('seoul.h5')
image = cv2.imread("./dataset/face/face_24.png")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image_gray.shape)
image_gray = image_gray.reshape(1, 160, 160, 1)
pred = model.predict(image_gray)


print(np.argmax(pred))



