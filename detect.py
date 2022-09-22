import cv2 as cv
import numpy as np

img = cv.imread('lady.jpg')
img = cv.imread(r'BangladeshiCelebrities\val\mehezabin/3.jpg')
cv.imshow('Lady in red', img)

gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade= cv.CascadeClassifier('haarcascade.xml')

face_rect= haar_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors= 2)

for(x,y,w,h) in face_rect:
    cv.rectangle(img,(x,y), (x+w, y+h), (255,0,255), thickness=3)

cv.imshow('detected face',img)
cv.waitKey(0)