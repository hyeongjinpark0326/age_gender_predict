import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import keras
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model #모델 도식 저장용
import tensorflow as tf
from adabelief_tf import AdaBeliefOptimizer


path = "C:/Users/admin/Desktop/final/images/utk_test/utk_resize_200/"
save_dir = os.path.join(os.getcwd(), 'age_gender_saved_models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_name = 'age_gender.h5'
model_path = os.path.join(save_dir, model_name)
pixels = []
age = []
gender = []

for img in os.listdir(path):
    ages = img.split("_")[0]
    genders = img.split("_")[1][0]
    # print(ages)
    # print(str(path)+str(img))
    img = cv2.imread(path+img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    pixels.append(np.array(img))
    age.append(np.array(ages))
    gender.append(np.array(genders))

age = np.array(age,dtype=np.int64)
pixels = np.array(pixels)
gender = np.array(gender,np.uint64)
# print(age)

x_train,x_test,y_train,y_test = train_test_split(pixels,age,random_state=100)
x_train_2,x_test_2,y_train_2,y_test_2 = train_test_split(pixels,gender,random_state=100)
input = Input(shape=(200,200,3))
conv1 = Conv2D(140,(7,7),activation="relu")(input)
conv2 = Conv2D(130,(5,5),activation="relu")(conv1)
batch1 = BatchNormalization()(conv2)
pool3 = MaxPool2D((2,2))(batch1)
conv3 = Conv2D(120,(5,5),activation="relu")(pool3)
batch2 = BatchNormalization()(conv3)
pool4 = MaxPool2D((2,2))(batch2)
conv4 = Conv2D(110,(3,3),activation="relu")(pool4)
batch3 = BatchNormalization()(conv4)
pool5 = MaxPool2D((2,2))(batch3)
flt = Flatten()(pool5)

#age neural network
age_l = Dense(128,activation="relu")(flt)
age_l = Dense(64,activation="relu")(age_l)
age_l = Dense(32,activation="relu")(age_l)
age_l = Dense(1,activation="relu")(age_l)

# #gender neural network
gender_l = Dense(128,activation="relu")(flt)
gender_l = Dense(80,activation="relu")(gender_l)
gender_l = Dense(64,activation="relu")(gender_l)
gender_l = Dense(32,activation="relu")(gender_l)
gender_l = Dropout(0.5)(gender_l)
gender_l = Dense(2,activation="softmax")(gender_l)

model = Model(inputs=input,outputs=[age_l, gender_l]) #출력형태를 2가지 형식으로
model.summary() #모델 구조 확인용

# plot_model(model, to_file='model_shapes.png', show_shapes=True) #모델 도식 저장파일

opt = AdaBeliefOptimizer(learning_rate=0.0001)
model.compile(optimizer=opt,
loss=["mse","sparse_categorical_crossentropy"], #integer type 클래스 -> one-hot encoding하지 않고 정수 형태로 label(y)을 넣어줌
metrics=['mae','accuracy']) #mse(mean squared error),mae(mean absolute error)
early_stopping = keras.callbacks.EarlyStopping(
monitor='val_loss', #val_loss가 더이상 감소되지 않을 경우
patience=10, #최적의 monitor 값을 기준으로 몇 번의 epoch을 진행할 지 정하는 값
verbose=1, #0일 경우, 화면에 나타냄 없이 종료합니다.
mode='auto') #"auto"는 모델이 알아서 판단합니다.
checkpoint = ModelCheckpoint(
filepath= save_dir + "/1112.h5",
monitor='val_loss', #validation set의 loss가 가장 작을 때 저장
verbose=1, #0일 경우 화면에 표시되는 것 없이 그냥 바로 모델이 저장됩니다.
save_best_only=True,
save_freq='epoch') #변경


save = model.fit(x_train,[y_train,y_train_2],
validation_data=(x_test,[y_test,y_test_2]),
epochs=200,
batch_size=8, #노트북 메모리 용량 초과로 줄임(16)
callbacks=[early_stopping,checkpoint])
#model.save(model_path)
#print(save.history.keys()) #key 확인용

#모델 학습 결과 그래프 표시
#loss 그래프
plt.figure(figsize=(17, 4)) #인치단위 크기
plt.subplot(1, 3, 1)
plt.plot(save.history['dense_3_loss'], label='dense_3_loss')
plt.plot(save.history['dense_8_loss'], label='dense_8_loss')
plt.plot(save.history['val_dense_3_loss'], label='val_dense_3_loss')
plt.plot(save.history['val_dense_8_loss'], label='val_dense_8_loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('loss VS Epochs')
plt.grid()
plt.legend()

#accuracy 그래프
plt.subplot(1, 3, 2)
plt.plot(save.history['dense_3_accuracy'], label='dense_3_accuracy')
plt.plot(save.history['dense_8_accuracy'], label='dense_8_accuracy')
plt.plot(save.history['val_dense_3_accuracy'], label='val_dense_3_accuracy')
plt.plot(save.history['val_dense_8_accuracy'], label='val_dense_8_accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.title('accuracy VS Epochs')
plt.grid()
plt.legend()

#MAE(Mean Absolute Error) 그래프
plt.subplot(1, 3, 3)
plt.plot(save.history['dense_3_mae'], label='dense_3_mae')
plt.plot(save.history['dense_8_mae'], label='dense_8_mae')
plt.plot(save.history['val_dense_3_mae'], label='val_dense_3_mae')
plt.plot(save.history['val_dense_8_mae'], label='val_dense_8_mae')
plt.xlabel('Epochs')
plt.ylabel('mae')
plt.title('mae VS Epochs')
plt.grid()
plt.legend()

plt.show()