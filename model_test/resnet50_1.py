from tensorflow.keras.applications import ResNet50, ResNet152
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 위치 경로
train_path="C:/Users/admin/Desktop/final/images/gray+resized/train/"
val_path="C:/Users/admin/Desktop/final/images/gray+resized/val/"

# number of classes
K = 2
input = Input(shape=(224, 224, 3))
model = ResNet152(input_tensor=input, include_top=False, weights=None, pooling='max')

x = model.output
x = Dense(1024, name='fully')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(512, )(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(K, activation='softmax', name='softmax')(x)
model = Model(model.input, x)
# model.summary()


train_datagen = ImageDataGenerator(rescale=1./255)#rgb값 reduce
train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=16,
        color_mode='rgb',
        class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=(224, 224),
        batch_size=16,
        color_mode='rgb',
        class_mode='categorical')

model.compile(loss='categorical_crossentropy',
                  #optimizer=optimizers.RMSprop(lr=2e-4),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

model.summary()

print_weights = LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[3].get_weights()))

es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
mc = ModelCheckpoint('./best_model_1005.h5', monitor='val_accuracy', mode='max')


history = model.fit(train_generator,
                              epochs=20,
                              validation_data=val_generator,                            
                              batch_size=100,
                              callbacks=[es, mc])


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
scores = model.evaluate_generator(val_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# json으로 저장
model_json = model.to_json()
with open("/content/gdrive/My Drive/resnet50.json","w") as json_file:
        json_file.write(model_json)
print("Saved model file to disk")
