from keras.preprocessing import image
import numpy as np
import os
from keras.models import Sequential
from keras.layers import *
from keras.layers.core import *
from keras import optimizers
from keras.callbacks import TensorBoard

size = 256
class_num = 2
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
model.load_weights("cnn_kabosu3.hdf5")

def pred(img):


    img = image.load_img(img, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

        # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要！
        # これを忘れると結果がおかしくなるので注意
    x = x/ 255.0
    kekka = np.argmax(model.predict(x))  #良品＝１、不良品＝０
    if kekka == 0:
        print("不良品")
    

    else:
        print("良品")
    return (kekka)

