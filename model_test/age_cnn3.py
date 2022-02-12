from keras.layers.core import Dropout
import matplotlib.pyplot as plt
import tensorflow  as tf
import tensorflow.keras as Keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import *
from tensorflow.keras.layers import Dense, Flatten, Input, LeakyReLU
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from keras.regularizers import l1, l2
from tensorflow.keras import regularizers, optimizers
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Conv2D, Flatten, MaxPooling2D
import cv2, os, re, glob
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
import keras

groups_folder_path = "C:/Users/admin/Desktop/final/images/last/Last_age_devided_200/"
categories = ["20", "30", "40", "50", "60"]
 
num_classes = len(categories)
  
image_w = 200
image_h = 200
  
X = []
Y = []
  
for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = groups_folder_path + categorie + '/'
  
    for top, dir, f in os.walk(image_dir):
        for filename in f:
            # print(image_dir+filename)
            img = cv2.imread(image_dir+filename)
            img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
            # print(img.shape)
            X.append(img/255)
            Y.append(label)
            
X = np.array(X)
Y = np.array(Y)
 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8, random_state=121)
# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)


# # 이미지 증강
# image_generator = ImageDataGenerator(
#     rotation_range=10, 
#     zoom_range=0.10, 
#     shear_range=0.3,        
#     width_shift_range=0.10, 
#     height_shift_range=0.10, 
#     horizontal_flip=True,
#     vertical_flip=False)

# # 증강할 이미지 개수
# augment_size = 8000

# randidx = np.random.randint(X_train.shape[0], size=augment_size)
# x_augmented = X_train[randidx].copy()
# y_augmented = Y_train[randidx].copy()
# x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size), batch_size=16, shuffle=False).next()[0]

# # 원래 데이터인 x_train에 이미지 보강된 x_augmented를 추가
# X_train = np.concatenate((X_train, x_augmented))
# Y_train = np.concatenate((Y_train, y_augmented))
# print(X_train.shape)
# print(Y_train.shape)


model = keras.models.Sequential([
Conv2D(100, kernel_size=(5,5), activation=LeakyReLU(0.2), input_shape=(image_w, image_h, 3), padding='same'),
Conv2D(100, (5,5), activation=LeakyReLU(0.2), padding='same'),
MaxPooling2D(pool_size=(2,2), strides = (2,2)),

Conv2D(150, (3,3), activation=LeakyReLU(0.2), padding='same'),
MaxPooling2D(pool_size=(2,2), strides = (2,2)),

Conv2D(200, (3,3), activation=LeakyReLU(0.2), padding='same'),
MaxPooling2D(pool_size=(2,2), strides=(2,2)),

# Conv2D(2048, (3,3), activation='relu', padding='same'),
# MaxPooling2D(pool_size=(2,2), strides=(2,2)),
# Dropout(0.5),

Flatten(),

# Dense(1024, activation='relu'),
# Dropout(0.2),
Dense(512, activation='relu'),
Dropout(0.2),
Dense(256, activation='relu'),
Dropout(0.2),
# Dense(128, activation='relu'),
# Dropout(0.2),
Dense(5, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

model.summary()

plot_model(model, to_file='age_cnn3_shapes.png', show_shapes=True)

cp_callback = callbacks.ModelCheckpoint(filepath="./gender_cnn3_1024.hdf5", 
                              monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
es_callback = callbacks.EarlyStopping(monitor='val_accuracy', 
                            mode='max', verbose=1, patience=20)    # patience=10, 20, 50

# Train CNN model

hist = model.fit(X_train, Y_train, epochs = 1000 , batch_size = 16, 
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