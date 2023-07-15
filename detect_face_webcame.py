import cv2
import numpy as np

my_video= cv2.VideoCapture(0) #0, çünkü görüntüyü webcam'den alıyorum.

cascade_file= cv2.CascadeClassifier("C:\OpenCV\\haarCascade\\facedet.xml") # cscade dosyamı include ediyorum

#her frame'i incelemek istiyorum

while 1:
    _,frame= my_video.read()
    frame=cv2.flip(frame,1) # y eksenine göre ters çeviriyorum

    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #haar'ı algılasın diye frame'leri griye çeviriyorum
    face_coordinates= cascade_file.detectMultiScale(gray, 1.4, 4)

    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)






