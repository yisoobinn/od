import tensorflow as tf 
import selectivesearch
import numpy as np 
import cv2

def calculate_iou(box1, box2):
    x1, y1, width1, height1 = box1
    x2, y2, width2, height2 = box2

    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + width1, x2 + width2)
    yB = min(y1 + height1, y2 + height2)

    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box1_area = width1 * height1
    box2_area = width2 * height2

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def non_max_suppression(boxes, scores, iou_threshold=0.7, max_output_size=100):
    selected_indices = []
    selected_scores = []

    for i in range(len(boxes)):
        keep = True
        for j in range(len(selected_indices)):
            iou = calculate_iou(boxes[i], boxes[selected_indices[j]])
            if iou > iou_threshold:
                keep = False
                break
        if keep:
            selected_indices.append(i)
            selected_scores.append(scores[i])

    return selected_indices, selected_scores

model = tf.keras.models.load_model('seoul.h5')
image_origin = cv2.imread("./dataset/wo_face/wo_face_5.png")

_, regions = selectivesearch.selective_search(image_origin, scale=5000, min_size=500)
image = image_origin.copy()
selected_boxes = []
selected_scores = []

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

            #image = cv2.putText(image, str(answer), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            #image = cv2.rectangle(image, (int(x1), int(y1)), (int(x1+width), int(y1+height)), color=(0, 0, 255), thickness=2) 

            selected_boxes.append(rect)
            selected_scores.append(answer)
    except:
        continue
print(selected_boxes)
selected_indices, selected_scores = non_max_suppression(selected_boxes, selected_scores)
print(selected_indices)
print(selected_scores)

for idx in selected_indices:
    try:
        x1, y1, width, height = selected_boxes[idx]
        answer = selected_scores[idx]
           
        image = cv2.putText(image, str(answer), (int(x1-10), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        image = cv2.rectangle(image, (int(x1), int(y1)), (int(x1+width), int(y1+height)), color=(0, 0, 255), thickness=2)
        
    except:
        continue
cv2.imshow("Candidate Regions", image)
cv2.waitKey(0) 


