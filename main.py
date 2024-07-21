import os

import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 680)  # Set width
cap.set(4, 340)  # Set height

segmentor = SelfiSegmentation()

imgBg = cv2.imread('images/bg2.jpeg')
listImage = os.listdir("images")

imgList = []
for imgPath in listImage:
    img = cv2.imread(f'images/{imgPath}')
    imgList.append(img)

indexImage = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # Resize background image to match the frame size
    imgBg = cv2.resize(imgList[indexImage], (img.shape[1], img.shape[0]))

    imgOut = segmentor.removeBG(img, imgBg, cutThreshold=0.9)

    imgStack = cvzone.stackImages([img, imgOut], 2, 1)
    cv2.imshow("imgStack", imgStack)

    key = cv2.waitKey(1)
    if key == ord('a'):
        if indexImage >0:
           indexImage -=1
        else:
            indexImage = len(imgList)-1

    if key == ord('d'):
        if indexImage <len(imgList)-1:
           indexImage +=1
        else:
            indexImage = 0

    if key == ord('q'):
        break

    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cap.release()
cv2.destroyAllWindows()
