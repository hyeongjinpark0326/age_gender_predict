from keras.layers.core import Dropout
import matplotlib.pyplot as plt
import tensorflow  as tf
from tensorflow.keras.layers import Input
import matplotlib.image as mpimg
import tensorflow.keras as Keras
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import *
from tensorflow.keras.layers import Dense, Flatten, GlobalMaxPooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
import efficientnet.keras as efn
from keras.regularizers import l1, l2
from tensorflow.keras import regularizers, optimizers
import os

# import keras.backend.tensorflow_backend as K
# with K.tf.device('/cpu:0'):



train_path="C:/Users/admin/Desktop/final/images/resized/train/"
val_path="C:/Users/admin/Desktop/final/images/resized/val/"

# Hyperparams
IMAGE_SIZE = 224
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
EPOCHS = 20000
BATCH_SIZE = 128 # generator 동작시 한번에 이미지 가져오는 양

input = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))

model = Sequential()

efficient_net = EfficientNetB1(weights='imagenet', include_top=False, input_tensor=input)

model.add(efficient_net)
model.add(GlobalMaxPooling2D())
model.add(Dense(1024, activation='relu'), )
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', name="fc1"))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax', name="output")) 
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
model.summary()



# Data augmentation
training_data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)
validation_data_generator = ImageDataGenerator(rescale=1./255)

# Data preparation
training_generator = training_data_generator.flow_from_directory(
    train_path,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical")

validation_generator = validation_data_generator.flow_from_directory(
    val_path,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical")


# collbacks
# tb = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
mc = ModelCheckpoint('./best_model_1006.h5', monitor='val_accuracy', mode='max')
cl = CSVLogger('./gender_efn_log', 'log.csv')
rlop = ReduceLROnPlateau()

# 학습률 lr 을 학습 세대의 횟수에 스케줄에 맞춰서 조절한다
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

lr = LearningRateScheduler( scheduler )



# Training
hist = model.fit(
    training_generator,
    batch_size = 30,
    epochs = 20,
    verbose = 1,
    validation_data=validation_generator,
    callbacks=[es, mc, cl,rlop])
    
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

