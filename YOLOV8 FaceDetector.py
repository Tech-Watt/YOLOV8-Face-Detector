import cvzone
from ultralytics import YOLO
import cv2




video = 'put the path to your video here'
cap = cv2.VideoCapture(video)

facemodel = YOLO('yolov8n-face.pt')


while cap.isOpened():
    rt, video = cap.read()
    video = cv2.resize(video, (700, 500))

    face_result = facemodel.predict(video,conf = 0.40)
    for info in face_result:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h,w = y2-y1,x2-x1
            cvzone.cornerRect(video,[x1,y1,w,h],l=9,rt=3)


    cv2.imshow('frame', video)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
