# https://www.kaggle.com/asdasdasasdas/garbage-classification
import tensorflow as tf
from keras_preprocessing.image import utils
from tensorflow.keras.applications import VGG16
import os, sys
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Flatten, Conv2D, Activation, MaxPool2D, Dropout
import matplotlib.pyplot as plt

label = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

opath = #your_directory_path
plist = os.listdir(opath)
print(plist)
dlist = []
llist = []

for i in plist:
    if i in label:
        idx = label.index(i)
    path = opath
    path = path + '/' + i
    dataset = glob.glob(os.path.join(path, '*.jpg'))
    for frame in dataset:
        img = Image.open(frame)
        temp = np.array(img)
        dlist.append(temp)

    for cnt in range(len(dataset)):
        llist.append(idx)

l_x_train = np.asarray(dlist)
print(l_x_train)
l_y_train = np.asarray(llist)

print(l_x_train.shape,len(l_x_train), l_y_train.shape, len(l_y_train))

seed = 0

np.random.seed(seed)
tf.random.set_seed(seed)

x_train, x_test, y_train, y_test = train_test_split(l_x_train, l_y_train, test_size=0.3, random_state=seed)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, 6)
y_test = to_categorical(y_test, 6)

# input = Input(shape=(128, 128, 3))
# model = VGG16(weights=None, include_top=False, input_tensor=input, pooling='None')
# x = model.output
# x = Flatten()(x)
# x = Dense(4096, activation='relu')(x)
# x = Dense(4096, activation='relu')(x)
# predictions = Dense(6, activation='softmax')(x)
# model = tf.keras.Model(inputs=model.input, outputs=predictions)

k_size = (3,3)
input = Input(shape=(128, 128, 3))
x = Conv2D(32, k_size, padding='same', strides=2)(input)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(64, k_size, padding='same', strides=1)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = MaxPool2D((2,2))(x)

x = Conv2D(128, k_size, padding='same', strides=2)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(256, k_size, padding='same', strides=2)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(512, k_size, padding='same', strides=1)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = MaxPool2D((2,2))(x)

x = Flatten()(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)

x = Dense(6, activation='softmax')(x)
model = tf.keras.Model(input, x)

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train, validation_data=(x_test, y_test),
                         epochs=100, batch_size=32, verbose=2)
