import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haarcascade.xml')

#person_faces = ['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling'] #for foreign 
person_faces = ['2018-2-60-046','2018-2-60-048','2019-1-60-024','2019-1-60-093','2019-1-60-094'] #for classmates
#person_faces = ['Siam Ahmed','Ziaul Hoque Polash','Nusrat Imroz Tisha','Tanjin Tisha','Afran Nisho','Shakib Al Hassan','Tahsan','Mehazabien Chowdhury','Sadiya Ayman','Nadia','Bidya Sinha Mim']

# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

lbph_recognizer = cv.face.LBPHFaceRecognizer_create()
lbph_recognizer.read('trained_faces.yml')

#img = cv.imread(r'ForeignCelebrities\val\ben_afflek/3.jpg') #for foreign celeb
img = cv.imread(r'classmates\val\riad/29.jpg') #for classmates
#img = cv.imread(r'classmates\val\riad/29.jpg') #for classmates
#img = cv.imread(r'BangladeshiCelebrities\val\tahsan/3.jpg') #for bangladeshi celeb
resized= cv.resize(img,(500,500))

gray_img = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
cv.imshow('gray image', gray_img)

# Detect the face in the image
rectangle_detect = haar_cascade.detectMultiScale(gray_img, 1.1, 4)

for (x,y,w,h) in rectangle_detect:
    gray_faces = gray_img[y:y+h,x:x+w]

    label, confidence = lbph_recognizer.predict(gray_faces)
    print(f'The person is {person_faces[label]} and the confidence level is: {confidence}')

    cv.putText(resized, str(person_faces[label]), (50,50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (155,0,255), thickness=2)
    cv.rectangle(resized, (x,y), (x+w,y+h), (155,0,255), thickness=2)

cv.imshow('Recognized Face', resized)
cv.waitKey(0)