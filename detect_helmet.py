import cv2
import numpy as np

img=cv2.imread("C:\\Users\Asus\Desktop\\barret.jpg")
img= cv2.resize(img,(480,360))
helmet_cascade= cv2.CascadeClassifier("C:\\Users\Asus\Desktop\isbaret\classifier\cascade.xml")


gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
helmet=helmet_cascade.detectMultiScale(gray,1.3,4)

for(x,y,w,h) in helmet:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4)


cv2.imshow("image",img)

cv2.waitKey(0)
cv2.destroyAllWindows()