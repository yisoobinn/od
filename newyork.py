import tensorflow as tf 
import selectivesearch
import numpy as np 
import cv2

model = tf.keras.models.load_model('seoul.h5')
image = cv2.imread("./dataset/face/face_24.png")

_, regions = selectivesearch.selective_search(image, scale=5000, min_size=500)

for cand in regions:
    try:
        length = cand['size']
        if length < 50000 and length > 100:
            rect = cand['rect']
            print(cand['size'])
            print("aa : {}, {}. {}. {} : ".format(rect[0], rect[1], rect[2], rect[3]))
  
            x1 =rect[0]
            y1 =rect[1]
            width = rect[2]
            height = rect[3]
            
            digit_region = image[int(y1):int(y1+height), int(x1):int(x1+width)]
            img_gray = cv2.cvtColor(digit_region, cv2.COLOR_BGR2GRAY)
            img_160 = cv2.resize(img_gray, (160, 160))

            train_x = img_160.reshape(1, 160, 160, 1)
            pred = model.predict(train_x)
            answer = np.argmax(pred)

            print(answer)

            image = cv2.putText(image, str(answer), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x1+width), int(y1+height)), color=(0, 0, 255), thickness=2) 
            cv2.imshow("Candidate Regions", image)
            cv2.waitKey(1) 
            input()
    except:
        continue

input()

