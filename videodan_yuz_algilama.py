import cv2

video= cv2.VideoCapture("C:\\Users\Asus\Desktop\\walking.mp4") #içinde yüz olan video ekliyorum

cascade_file= cv2.CascadeClassifier("C:\OpenCV\\haarCascade\\facedet.xml") #cascade dosyamı include ediyorum.

while 1: #sonsuz döngü oluşturuyorum
    _,frame= video.read() #videodaki her bir frame'i gözlemliyorum
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = cascade_file.detectMultiScale(gray,1.3,4)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("face",frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

video.release() #videoyu serbest bırakıyorum
cv2.destroyAllWindows()