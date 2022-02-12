import tensorflow as tf
import tensorflow.keras as Keras
from tensorflow.keras import models, layers, Model
from tensorflow.keras import Input
from keras.models import Sequential
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, initializers, regularizers, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Flatten
from tensorflow.keras.applications import ResNet50
import os, cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.svm import SVC
from sklearn import svm
import keras

train_path = 'C:/Users/admin/Desktop/final/images/last/age_combined_gray_resized_copy_140/'
categories = ["20", "30", "40", "50", "60"]

 
num_classes = len(categories)
  
image_w = 140
image_h = 140
  
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

model = keras.models.Sequential([
    
    Conv2D(input_shape=(image_w, image_h,3),
          kernel_size=(9,9),  strides=(2,2),filters= 96,padding='same',activation='relu'),
    MaxPooling2D(pool_size=(3, 3)),
    BatchNormalization(),
                                                                
    Conv2D(input_shape=(image_w, image_h,3),
          kernel_size=(7,7),  strides=(2,2), filters= 256,padding='same',activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(kernel_size=(3,3),filters= 256, strides=(2,2), padding='same',activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(), 
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024, activation='relu', name="fc1", kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)),  
    keras.layers.Dropout(0.5),
    keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)),  
    # keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)),  
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)),  
    keras.layers.Dense(5, activation='softmax')])

model.compile(loss='categorical_crossentropy',
                  #optimizer=optimizers.RMSprop(lr=2e-4),
                  optimizer = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

model.summary()

#print_weights = LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[3].get_weights()))

def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

# es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
mc = ModelCheckpoint('.age_resnet50_1023.h5', monitor='val_accuracy', mode='max')
lr = LearningRateScheduler( scheduler )
rlp = ReduceLROnPlateau(
    monitor='val_accuracy',  #
    factor=0.2,          
    patience=20,
)


#with tf.device('/device:GPU:0'):    
history = model.fit(X_train, Y_train,
                        epochs = 1000,
                        validation_data=(X_test, Y_test),                            
                        batch_size = 16,
                        callbacks=[mc,lr ,rlp])
#모델 학습 결과 그래프 표시
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b--', label='loss')
plt.plot(history.history['val_loss'], 'r:', label='val_loss')
plt.xlabel('Epochs')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b--', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r:', label='val_accuracy')
plt.xlabel('Epochs')
plt.grid()
plt.legend()

plt.show()

#모델 평가
print("-- Evaluate --")
scores = model.evaluate([X_test, Y_test], steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))