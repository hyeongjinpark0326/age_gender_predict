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
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D, LeakyReLU, Flatten, Input, Dropout
from adabelief_tf import AdaBeliefOptimizer
from tensorflow.python.keras.applications.efficientnet import EfficientNetB1

# C:/Users/admin/Desktop/final/images/last/age_combined_gray_resized_copy_140/
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

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8, random_state=64)
print(X_train.shape)
print(Y_train.shape)

input = Input(shape=(image_w, image_h, 3))

model = Sequential()

efficient_net = EfficientNetB1(weights='imagenet', include_top=False, input_shape=input)
efficient_net.trainable = False

model.add(efficient_net)
# # model.summary()
model.add(Flatten())
model.add(Dense(1024))
# #model.add(LeakyReLU(0.5))
model.add(Dropout(0.5))
model.add(Dense(512,name="fc1"))
# #model.add(LeakyReLU(0.5))
model.add(Dropout(0.5))
model.add(Dense(128))
# #model.add(LeakyReLU(0.5)) model.add(Dropout(0.2))
model.add(Dense(9, activation='softmax', name="output")) 

opt = AdaBeliefOptimizer(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy',
             optimizer=opt, #learning_rate=0.0001, decay=0.0005
            metrics=['accuracy'])
model.summary()


# collbacks
# tb = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
mc = ModelCheckpoint('./efn_age_1106.h5', monitor='val_accuracy', mode='max')
#cl = CSVLogger('./CSVLogger', 'log.csv')

# Training
#with tf.device('/device:GPU:0'):    
hist = model.fit(
    X_train, Y_train, epochs = 5000 , batch_size = 6, 
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

model.save('age_efn_1106.h5')