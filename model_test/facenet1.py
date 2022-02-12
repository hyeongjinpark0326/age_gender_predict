import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
import cv2, os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from keras.layers import Conv2D, Flatten, MaxPooling2D
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

K = 5

groups_folder_path = "C:/Users/admin/Desktop/final/images/age_combined_gray_resized/"
categories = ["20", "30", "40", "50", "60"]
 
num_classes = len(categories)
  
image_w = 146
image_h = 146
  
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
            X.append(img/256)
            Y.append(label)
            
X = np.array(X)
Y = np.array(Y)
 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8, random_state=121)
print(X_train.shape)
print(Y_train.shape)


facenet_model = tf.keras.models.load_model('facenet_keras.h5',)  

for layer in facenet_model.layers:
    layer.trainable = False

input = facenet_model.input
#input = facenet_model.layers[0].output
output = facenet_model.layers[-1].output
output = tf.keras.layers.Dense(units=32, activation='relu')(output)
output = tf.keras.layers.Dense(K, activation='softmax')(output)
model = tf.keras.models.Model(input, output)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# callbacks
cp_callback = callbacks.ModelCheckpoint(filepath="./facenet1_1015.hdf5", 
                              monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
es_callback = callbacks.EarlyStopping(monitor='val_accuracy', 
                            mode='max', verbose=1, patience=20)    # patience=10, 20, 50

# Train CNN model

hist = model.fit(X_train, Y_train, epochs = 5000 , batch_size = 1000, 
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