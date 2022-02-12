from keras.layers.core import Dropout
import matplotlib.pyplot as plt
import tensorflow  as tf
from tensorflow.keras.layers import Input
import matplotlib.image as mpimg
import tensorflow.keras as Keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalMaxPooling2D, Input, Activation
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import regularizers, optimizers
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPool2D, Flatten, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
import os, re, glob, cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2


groups_folder_path = "C:/Users/admin/Desktop/final/images/age_combined_gray_resized_copy_140/"
categories = ["20", "30", "40", "50", "60"]
 
num_classes = len(categories)
  
image_w = 140
image_h = 140
  
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
            # img = cv2.imwrite('./ab1.jpg', img)
            X.append(img/256)
            Y.append(label)
            
X = np.array(X)
Y = np.array(Y)
 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8, random_state=121)
print(X_train.shape)

K = 5
input = Input(shape=(140, 140, 3))
model = Sequential()
isv2 = InceptionResNetV2(input_tensor=input, include_top=False, weights=None, pooling='max')
model.trainable = False

model.add(isv2)
# model.summary()

x = model.output
x = Dense(1024, name='fully')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(512, )(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(128, )(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(K, activation='softmax', name='softmax')(x)
model = Model(model.input, x)
model.summary()



# model.compile(loss='categorical_crossentropy',
#                   #optimizer=optimizers.RMSprop(lr=2e-4),
#                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                   metrics=['accuracy'])

# model.summary()

# # print_weights = LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[3].get_weights()))

# es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
# mc = ModelCheckpoint('./best_model_1005.h5', monitor='val_accuracy', mode='max')


# history = model.fit(X_train, Y_train,
#                               epochs=20,
#                               validation_data=[X_test, Y_test],                            
#                               batch_size=32,
#                               callbacks=[es, mc])


# #모델 학습 결과 그래프 표시
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], 'b--', label='loss')
# plt.plot(history.history['val_loss'], 'r:', label='val_loss')
# plt.xlabel('Epochs')
# plt.grid()
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], 'b--', label='accuracy')
# plt.plot(history.history['val_accuracy'], 'r:', label='val_accuracy')
# plt.xlabel('Epochs')
# plt.grid()
# plt.legend()

# plt.show()

# #모델 평가
# print("-- Evaluate --")

# scores = model.evaluate_generator([Y_train, Y_test], steps=5)
# print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# # # json으로 저장
# # model_json = model.to_json()
# # with open("/content/gdrive/My Drive/resnet50.json","w") as json_file:
# #         json_file.write(model_json)
# # print("Saved model file to disk")
