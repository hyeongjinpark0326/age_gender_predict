import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2, os
import dlib




# model = tf.keras.models.load_model()

age_model_path = "age_gender_1105_models.h5"
gender_model_path = "C:/Users/admin/Desktop/1108.h5"
model1 = load_model(age_model_path) 
model2 = load_model(gender_model_path)
output_path = "C:/work/realtime_age_gender-main/predict_results/"
img_path = "C:/Users/admin/Desktop/aaaa/"
face_cascade = cv2.CascadeClassifier("C:/Users/admin/Desktop/final/age_gender_estimation-master (2)/haarcascade_frontalface_default.xml")
for img in os.listdir(img_path):
 
  pic = cv2.imread('C:/Users/admin/Desktop/233.jpg')
  gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(pic,scaleFactor=1.11, minNeighbors=8)
  # print(faces)

  age_ = []
  gender_ = []
  for (x,y,w,h) in faces:
    img = pic[y:y + h, x:x + w]
    # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img1 = cv2.resize(img,(200,200))
    predict1 = model1.predict(np.array(img1).reshape(-1,200,200,3))  
    # print(predict1)
    print('++++++++++++++++++++++++++ 예측 시작 1 ++++++++++++++++++++++')
    img2 = cv2.resize(img,(200, 200))
    predict2 = model2.predict(np.array(img2).reshape(-1,200,200,3))
    print('++++++++++++++++++++++++++ 예측 시작 2 ++++++++++++++++++++++')
    age_.append(predict1[0])
    print( predict1 )
    print( predict2 )
    gender_.append(np.argmax(predict2[0]))
    print(predict2)
    gend = np.argmax(predict2[0])
    if gend == 0:
      gend = 'Man'
      col = (255,255,0)
    else:
      gend = 'Woman'
      col = (203,12,255)
    cv2.rectangle(pic,(x,y),(x+w,y+h),(0,225,0),5)
    cv2.putText(pic,"Age : "+str(int(predict1[0]))+" / "+str(gend),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,w*0.005,col,1)
  pic1 = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
  # print(age_,gender_)
  plt.imshow(pic1)
  plt.show()
  cv2.waitKey(0)
  
