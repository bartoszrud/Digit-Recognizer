import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D


height,width = 28, 28
number_of_classes = 10
# (x_train, y_train), (x_test,y_test) = minst.load_data()

def data_preparation(data):
    y = keras.utils.to_categorical(data.label, number_of_classes)
    num_images = data.shape[0]
    x_array = data.values[:,1:]
    x_shaped_array = x_array.reshape(num_images,height,width,1)
    x = x_shaped_array/255
    return x,y


sciezka_danych_trening ='kaggle\\train.csv'
dane_trening = pd.read_csv(sciezka_danych_trening)
x_train,y_train = data_preparation(dane_trening)


model = tf.keras.models.Sequential()

model.add(Conv2D(64,kernel_size=(3,3),activation = 'elu', input_shape=(height,width,1)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation = 'elu'))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation = 'elu'))
model.add(MaxPooling2D(pool_size=(2,2), strides =(2,2)))
model.add(Conv2D(64,(3,3),activation = 'elu'))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation = 'elu'))



model.add(tf.keras.layers.Flatten())
model.add(Dense(256,activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(256,activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(number_of_classes,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=128,epochs =16, validation_split=0.2)


model.save('test.model')
# model.summary()
