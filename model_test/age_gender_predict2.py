from PIL import Image
import face_recognition
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt


age_model_path = "C:/Users/admin/Desktop/final/age_gender_estimation-master (2)/age_gender_1105_models.h5"
gender_model_path = "C:/Users/admin/Desktop/final/age_gender_estimation-master (2)/gender_1019.hdf5"
model1 = load_model(age_model_path) 
model2 = load_model(gender_model_path)

image = "C:/Users/admin/Desktop/final/age_gender_estimation-master (2)/abc.jpg"

imgElon = face_recognition.load_image_file("C:/Users/admin/Desktop/final/age_gender_estimation-master (2)/asfawfadsf.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)


faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# print(faceLoc)

face = []
faceLoc = list(faceLoc)
face.append(faceLoc)
print(faceLoc)

age_ = []
gender_ = []
for x,y,w,h in faceLoc:
    print(x)
    img = imgElon[y:y + h, x:x + w]
    # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img1 = cv2.resize(img,(200,200))
    predict1 = model1.predict(np.array(img1).reshape(-1,200,200,3))  
    print(predict1)
    print('++++++++++++++++++++++++++ 예측 시작 1 ++++++++++++++++++++++')
    img2 = cv2.resize(img,(140, 140))
    predict2 = model2.predict(np.array(img2).reshape(-1,200,200,3))
    print('++++++++++++++++++++++++++ 예측 시작 2 ++++++++++++++++++++++')
    age_.append(predict1[0])
    print( predict2 )
    cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 0), 2)


    cv2.imshow('Elon Musk', imgElon)
    cv2.waitKey(0)