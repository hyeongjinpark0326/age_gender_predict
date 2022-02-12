import numpy as np
import pandas as pd
import keras
import cv2
import os, re, glob
import tensorflow.keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import layers, models, callbacks, regularizers, optimizers
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, Conv3D, MaxPool2D, Flatten, AveragePooling2D, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

train_path="C:/Users/admin/Desktop/final/images/age_combined_gray_resized/"
categories = ["20", "30", "40", "50", "60" ]
 
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
            # print(image_dir+filename)
            img = cv2.imread(image_dir+filename)
            # print(image_w/img.shape[1], image_w/img.shape[0])
            img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
            #print(img.shape[1], img.shape[0])
            X.append(img/256)
            Y.append(label)
            
X = np.array(X)
Y = np.array(Y)
 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8, random_state=121)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

image_generator = ImageDataGenerator(
    rotation_range=10, 
    zoom_range=0.10, 
    shear_range=0.5, 
    width_shift_range=0.10, 
    height_shift_range=0.10, 
    horizontal_flip=True,
    vertical_flip=False)

augment_size = 1000

randidx = np.random.randint(X_train.shape[0], size=augment_size)
x_augmented = X_train[randidx].copy()
y_augmented = Y_train[randidx].copy()
x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[0]

# 원래 데이터인 x_train에 이미지 보강된 x_augmented를 추가합니다. 
X_train = np.concatenate((X_train, x_augmented))
Y_train = np.concatenate((Y_train, y_augmented))


model = keras.models.Sequential([
    Conv2D(input_shape=(224,224,3),kernel_size=(3,3), strides=1, filters= 32,padding='same',activation='relu'),
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
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)),  
    keras.layers.Dropout(0.7),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


cp_callback = callbacks.ModelCheckpoint(filepath="./age_cnn2.hdf5", 
                              monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
es_callback = callbacks.EarlyStopping(monitor='val_accuracy', 
                            mode='max', verbose=1, patience=20)    # patience=10, 20, 50

# Train CNN model
hist = model.fit(X_train, Y_train, epochs = 20000 , batch_size =64, 
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