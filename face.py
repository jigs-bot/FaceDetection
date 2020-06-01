import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


img = cv.imread('birthday.jpg')


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray, 1.1, 4)


for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Output
small=cv.resize(img,(0,0), fx=0.3, fy=0.3)
cv.imshow('img', small)
cv.waitKey()
