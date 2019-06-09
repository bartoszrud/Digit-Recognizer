import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

height ,width = 28,28
test_data_path ='kaggle\\test.csv'
test_data = pd.read_csv(test_data_path)

model = tf.keras.models.load_model('test.model')

def test_data_preparation(data):
    num_images = data.shape[0] # taking number of images
    x_array = data.values[:,0:]
    x_shaped_array = x_array.reshape(num_images,height,width,1)
    x_testowe = x_shaped_array/255
    return x_testowe


x_test = test_data_preparation(test_data)

predictions = model.predict_classes([x_test])
# print(predictions[0:5])


pred_df = pd.DataFrame({'Label': predictions })
pred_df.index+=1

pred_df.to_csv('kaggle_comp_hwd_9916.csv')
print(pred_df.head())

model.summary()
