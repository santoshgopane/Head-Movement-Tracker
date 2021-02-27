import cv2
import numpy as np
from mtcnn import MTCNN

cap = cv2.VideoCapture(0)

_,frame = cap.read()
old_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

detector1 = MTCNN()

lk_params = dict(winSize = (20,20),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.3))

while True:
    
    _ , image = cap.read()
    gray_frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img = image.copy()
    img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_arr = np.array(img_rgb)
    result = detector1.detect_faces(image_arr)
    try:
        nose = result[0]['keypoints']['nose']
        old_points = np.array([nose],dtype=np.float32) #Keeping it float32 is IMPORTANT!
        # print(old_points)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        # print(error)
        old_gray = gray_frame.copy()
        old_points = new_points

        X1,Y1 = new_points.ravel()
        cv2.circle(image,(X1,Y1),5,(0,255,0),2)

        bounding_box = result[0]['box']
        cv2.rectangle(img,(bounding_box[0],bounding_box[1]),
                  (bounding_box[0]+bounding_box[2],bounding_box[1]+bounding_box[3]),
                 (255,0,0),2)
        cv2.putText(img, 'Santosh', (bounding_box[0],bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Head Movement Tracking',image)
        cv2.imshow('Face Detection',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        cv2.imshow('Head Movement Tracking',image)
        cv2.imshow('Face Detection',img)
        print("You're out of Frame!")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()