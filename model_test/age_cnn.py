from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, Conv3D, MaxPool2D, Flatten, AveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l2
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.optimizers import Adam
import cv2
import os, re, glob

train_path = "C:/Users/admin/Desktop/dddd/"
categories = ["20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-"]
 
num_classes = len(categories)
  
image_w = 200
image_h = 200
  
X = []
Y = []
  
for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = train_path + categorie + '/'
  
    for top, dir, f in os.walk(image_dir):
        for filename in f:
            # print(filename)
            # print(image_dir+filename)
            img = cv2.imread(image_dir+filename)
            img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
            # img = cv2.imwrite('./ab1.jpg', img)
            X.append(img/256)
            Y.append(label)
            
X = np.array(X)
Y = np.array(Y)
 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8, random_state=121)
# print(X_train.shape)

model = keras.models.Sequential([
    Conv2D(input_shape=(image_w, image_h,3),kernel_size=(3,3), strides=1, filters= 32,padding='same',activation='relu'),
    AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),

    Conv2D(kernel_size=(5,5),strides=(1,1),filters= 64,padding='same',activation='relu'),
    MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

    Conv2D(kernel_size=(3,3),strides=(1,1),filters= 128,padding='same',activation='relu'),
    MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    
    Flatten(), 
    keras.layers.Dense(1024, activation='relu'),  
    keras.layers.Dropout(0.7),
    keras.layers.Dense(512, activation='relu'),  
    keras.layers.Dropout(0.7),
    keras.layers.Dense(128, activation='relu'),  
    keras.layers.Dropout(0.5),
    keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001, decay=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


cp_callback = callbacks.ModelCheckpoint(filepath="./age_cnn_1106.hdf5", 
                              monitor='val_accuracy', verbose=0, save_best_only=True)
es_callback = callbacks.EarlyStopping(monitor='val_accuracy', 
                            mode='max', verbose=1, patience=20)    # patience=10, 20, 50

# Train CNN model

hist = model.fit(X_train, Y_train, epochs = 5000 , batch_size = 16, 
         callbacks=[cp_callback, es_callback], 
         validation_data=(X_test,Y_test))

#모델 학습 결과 그래프 표시
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