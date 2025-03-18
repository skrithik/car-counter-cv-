import numpy as np
from sympy.utilities.codegen import Result
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


#uploading video
cap=cv2.VideoCapture("../yolo with webcam/videos/cars.mp4")
#weights of yolo alogorithm
model=YOLO('../YOLOwights/yolov8n.pt')
#class names for different objects which we want to detect
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
#create a mask for required region where we want to detect
#mask=cv2.imread('mask.png')

#tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

#creating a line after which when an id crosses it we would increment the counter
limits = [400,297,673,297]

#initializing count as 0(no of vehicles crossed the line)
totalCount=[]

#
while True :
    success , image= cap.read()
    #applying and on images so that only the part which is unmasked would be visible
    #imageRegion=cv2.bitwise_and(image,mask)
    results = model(image,stream=True)

    #adding the image of a car showing car count
    imgGraphics=cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED)
    image=cvzone.overlayPNG(image,imgGraphics,(0,0))
    #setting detections for tracker
    detections = np.empty((0,5))
    for r in results:
        boxes=r.boxes
        for box in boxes:
            #bounding box
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            #cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,255),3)

            w=x2-x1
            h=y2-y1

            #confidence
            conf=math.ceil((box.conf[0]*100))/100
            #cvzone.putTextRect(image,f'{conf}',(max(0,x1),max(35,y1)))
            #class
            cls=int(box.cls[0])
            currentClass = classNames[cls]
            #we use if condition if we want to display only the required class or desired class
            if (currentClass in ['car', 'bus', 'motorbike', 'truck']) and conf > 0.3:
                #making a detections variable of numpy stack type which we will send to tracker
                currentArray=([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))
    #a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]-using update method
    resultsTracker = tracker.update(detections)
    #creating line which would help in counting
    cv2.line(image, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    #going through the resultsTracker
    for result in resultsTracker:

        x1, y1, x2, y2, Id = map(int, result)
        w, h = x2 - x1, y2 - y1  # Fix: Calculate width and height

        # Draw Tracking Bounding Box
        cvzone.cornerRect(image, (x1, y1, w, h), l=9, rt=5, colorR=(255, 0, 255))

        # Display Object ID
        cvzone.putTextRect(image, f'ID: {int(Id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=5)

        #setting centre of boxes to identify whenever centre crosses the line count should be incremented
        cx,cy=x1+w//2,y1+h//2
        #displaying the centre
        cv2.circle(image,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limits[0]<cx<limits[2] and limits[1]-15 < cy < limits[1]+15:
            #we append id only once if we dont do this same car gets counted three times because
            #the centre may be in limits in multiple frames. So , we have to do this to avoid multiple counting
            if totalCount.count(Id)==0:
                totalCount.append(Id)
                cv2.line(image, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cv2.putText(image, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    cv2.imshow("Image",image)
    #if we want the masked region use imageRegion instead of image
    cv2.waitKey(1)
