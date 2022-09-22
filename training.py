import os
import cv2 as cv
import numpy as np

#person_faces = ['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling'] #for foreign 
#DIR = r'ForeignCelebrities\train' #for foreign celeb

person_faces = ['2018-2-60-046','2018-2-60-048','2019-1-60-024','2019-1-60-093','2019-1-60-094'] #for classmates
DIR = r'classmates\train' #for classmates

#person_faces = ['2019-1-60-005','2019-1-60-023','2019-1-60-055','2019-1-60-060','2019-1-60-075','2019-1-60-098','2019-1-60-166','2019-1-60-171','2019-1-60-172','2019-1-60-173','2019-1-60-204'] #for bd celeb
#DIR = r'BangladeshiCelebrities\train' #for bd celeb

haar_cascade = cv.CascadeClassifier('haarcascade.xml')

object = []
labeling = []

#function for training
def training():
    for faces in person_faces:
        path = os.path.join(DIR, faces)
        label = person_faces.index(faces)

        for image in os.listdir(path):
            directory = os.path.join(path,image)

            img_path = cv.imread(directory)
            if img_path is None:
                continue 
                
            gray = cv.cvtColor(img_path, cv.COLOR_BGR2GRAY) #to convert BGR image on grayscale image
            rectangle_detect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

            for (x,y,w,h) in rectangle_detect:
                gray_faces = gray[y:y+h, x:x+w]
                object.append(gray_faces)
                labeling.append(label)

training()
print('Trained all images successfully!!!')

feature_obj = np.array(object, dtype='object')
new_label = np.array(labeling)

face_recog = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recog.train(feature_obj,new_label)

face_recog.save('trained_faces.yml')
np.save('img_features.npy', feature_obj)
np.save('labeling.npy', new_label)