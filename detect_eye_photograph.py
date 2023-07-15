import cv2
import numpy as np

img= cv2.imread("C:\\Users\Asus\Desktop\yildiz.jpg")

cascade_file= cv2.CascadeClassifier("C:\OpenCV\\haarCascade\\eye.xml") # kullandığım haar kaynağı:  "https://github.com/opencv/opencv/tree/master/data/haarcascades"

gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

eye=cascade_file.detectMultiScale(gray,1.3,4)

for(x,y,w,h) in eye:
    cv2.rectangle(img,(x,y),(x+w,y+h),(130,100,100),2)

img2= img[y:y+h,x:x+w]
gray2=gray[y:y+h,x:x+w]

eyes= cascade_file.detectMultiScale(gray2)

for(ex,ey,ew,eh) in eyes:
    cv2.rectangle(img2,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

cv2.imshow("Yildiz Tilbe", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
