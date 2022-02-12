import matplotlib.pyplot as plt
import os, re, glob, cv2
import numpy as np
import pandas as pd
import tensorflow  as tf
import tensorflow.keras as Keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import regularizers, optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D, Flatten, Input, Dropout
from adabelief_tf import AdaBeliefOptimizer
from tensorflow.keras.optimizers import Adam, SGD

groups_folder_path = path = "C:/Users/admin/Desktop/abc/"
categories = ["20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-60"]
 
num_classes = len(categories)
  
image_w = 140
image_h = 140
  
X = []
Y = []
  
for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = groups_folder_path + categorie + '/'
    # print(label)
    for top, dir, f in os.walk(image_dir):
        for filename in f:
            # print(image_dir+filename)
            img = cv2.imread(image_dir+filename)
            img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
            X.append(img/256)
            Y.append(label)
            
X = np.array(X)
Y = np.array(Y)
 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8, random_state=121)
# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)



# # Data augmentation
# training_data_generator = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True)
# validation_data_generator = ImageDataGenerator(rescale=1./255)

# augment_size = 1000

# randidx = np.random.randint(X_train.shape[0], size=augment_size)
# x_augmented = X_train[randidx].copy()
# y_augmented = Y_train[randidx].copy()
# x_augmented = training_data_generator.flow(x_augmented, np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[0]

# # 원래 데이터인 x_train에 이미지 보강된 x_augmented를 추가합니다. 
# X_train = np.concatenate((X_train, x_augmented))
# Y_train = np.concatenate((Y_train, y_augmented))
# # print(X_train)
# # print(Y_train)


input = Input(shape=(140, 140, 3))

model = Sequential()

efficient_net = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_tensor=input)
efficient_net.trainable = False

model.add(efficient_net)
model.add(GlobalMaxPooling2D())
model.add(Dense(256, activation='relu', name="fc1"))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='softmax', name="output")) 

#opt = AdaBeliefOptimizer(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy',
            optimizer=SGD(learning_rate=0.0001, momentum=0.9, nesterov=True), #learning_rate=0.0001, decay=0.0005
            metrics=['accuracy'])
model.summary()



# collbacks
# tb = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
mc = ModelCheckpoint('./efn_age_1111.h5', monitor='val_accuracy', mode='max')
#cl = CSVLogger('./CSVLogger', 'log.csv')

# Training
#with tf.device('/device:GPU:0'):    
hist = model.fit(
    X_train, Y_train, epochs = 200 , batch_size = 8, 
        callbacks=[es, mc], 
        validation_data=(X_test,Y_test))
    
# 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], 'b--', label='loss')
plt.plot(hist.history['val_loss'], 'r:', label='val_loss')
plt.xlabel('Epochs')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist.history['accuracy'], 'b--', label='accuracy')
plt.plot(hist.history['val_accuracy'], 'r:', label='val_accuracy')
plt.xlabel('Epochs')
plt.grid()
plt.legend()
plt.show()
