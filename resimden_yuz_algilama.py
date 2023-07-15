import cv2

img= cv2.imread("C:\\Users\Asus\Desktop\\turkansoray.jpg")
img=cv2.resize(img,(500,500), interpolation=cv2.INTER_AREA) #yeniden boyutlandırıyorum
face_cascade= cv2.CascadeClassifier('C:\OpenCV\haarCascade\\facedetection.xml')

gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray,1.3,4)#gri, 1.3 değerinde küçült, en az 4 pencere yüz bulsun

for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3) #kalınlığı 3 olsun

cv2.imshow("Turkan Soray",img)

cv2.waitKey(0)
cv2.destroyAllWindows()

