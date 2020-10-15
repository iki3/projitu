from keras.preprocessing import image
import numpy as np
import os
from keras.models import Sequential
from keras.layers import *
from keras.layers.core import *
from keras import optimizers
import keras.callbacks
from keras.callbacks import TensorBoard
size = 256
train_num=1112
test_num = 40
train_batchsize=32
test_batchsize=test_num
train_dir = "train_images"
validation_dir = "test_images"
test_dir = "test_images"

classes=["bad","good"]
class_num=len(classes)
train_datagen = image.ImageDataGenerator(rescale=1./255.)
train_generator=train_datagen.flow_from_directory(
     train_dir,                      # ターゲットディレクトリ
    target_size=(size,size),          # すべての画像サイズを120*40に変更
    color_mode='rgb',     # ここを追加
    batch_size=32,  
    shuffle= True,
    classes = classes,# バッチサイズ
    class_mode='categorical',
    
)

test_datagen = image.ImageDataGenerator(rescale=1./255.)
validation_generator=train_datagen.flow_from_directory(
     test_dir,                      # ターゲットディレクトリ
    target_size=(size,size),          # すべての画像サイズを120*40に変更
    color_mode='rgb',     # ここを追加
    batch_size=32,  
    shuffle= True,
    classes = classes,# バッチサイズ
    class_mode='categorical',
)

model=Sequential()
model.add(Conv2D(32,(5,5),padding="same",activation="relu",input_shape=(size,size,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64,(5,5),padding="same",activation="relu",input_shape=(size,size,3)))
model.add(MaxPool2D(pool_size=(2,2)))
#model.add(Conv2D(128,(5,5),padding="same",activation="relu",input_shape=(size,size,3)))
#model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64,activation = "relu"))
model.add(Dense(class_num,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer=optimizers.SGD(lr=0.001,momentum=0.9),metrics=["accuracy"])

tb_cb = TensorBoard(log_dir="./logs")

history = model.fit_generator(
train_generator,
steps_per_epoch=train_num//train_batchsize,
validation_data=validation_generator,
validation_steps=test_num//test_batchsize,
epochs=50,
)

model.save_weights("cnn_kabosu3.hdf5")

