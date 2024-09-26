'''
This Code implements a basic CNN regression model with hyperparamter sweep 
'''
# Import Libraries
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_absolute_percentage_error
import random
import os
import importlib
import keras_tuner as kt
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Create Dataset
csv_test_loc="./Test_full_set.csv"
img_w,img_h=200,200
dftest=pd.read_csv(csv_test_loc)
dftest=dftest.to_numpy()
condition=dftest[:,1]>15
dftest=dftest[condition]
amounttest=len(dftest)



test_data={'Image': (np.array(dftest[:,3]).astype('U')),
            'Heat Flux': (np.array(dftest[:,1]).astype('float'))}


    
@tf.function
def load_and_preprocess_image(path):
    image_string = tf.io.read_file(path)
    img = tf.io.decode_png(image_string, channels=1)
    img = tf.image.resize(img, [img_w, img_h])
    img = tf.cast(img, tf.float32) / 255.0 
    return img

def load_and_preprocess_from_path_labels(image_path, labels):
    image = load_and_preprocess_image(image_path)
    return image, labels

def configure_for_performance(ds):
    ds = ds.cache()
    
    ds = ds.batch(50)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

test_ds = tf.data.Dataset.from_generator(lambda: zip(test_data['Image'], test_data['Heat Flux']), (tf.string, tf.float32))
test_ds = test_ds.map(load_and_preprocess_from_path_labels, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = configure_for_performance(test_ds)



for s1,s3 in test_ds.take(1):
    print("Image shape: ", s1.numpy().shape)
    print("Label: ", s3.numpy().shape)


# Define Model
def model_def():
	input_=tf.keras.layers.Input(shape=(200,200,1))
	hp_activation='relu'
	hp_layer1_filter=62
	hp_layer1_kernel=3
	hp_layer1_stride=1
	x=tf.keras.layers.Conv2D(hp_layer1_filter, hp_layer1_kernel,hp_layer1_stride, activation=hp_activation)(input_)
	hp_layer2_kernel=6
	hp_layer2_stride=3
	x=tf.keras.layers.MaxPool2D(hp_layer2_kernel,hp_layer2_stride)(x)
	x=tf.keras.layers.Dropout(0.2)(x)
	hp_layer3_filter=162
	hp_layer3_kernel=5
	hp_layer3_stride=1
	x=tf.keras.layers.Conv2D(hp_layer3_filter, hp_layer3_kernel,hp_layer3_stride, activation=hp_activation)(x)
	hp_layer4_kernel=4
	hp_layer4_stride=3
	x=tf.keras.layers.MaxPool2D(hp_layer4_kernel, hp_layer4_stride)(x)
	hp_layer5_filter=282
	hp_layer5_kernel=3
	hp_layer5_stride=3
	x=tf.keras.layers.Conv2D(hp_layer5_filter, hp_layer5_kernel,hp_layer5_stride, activation=hp_activation)(x)
	x=tf.keras.layers.GlobalAveragePooling2D()(x)
	x=tf.keras.layers.Dropout(0.2)(x)
	hp_activation1='relu'
	hp_layer6=501
	x=tf.keras.layers.Dense(hp_layer6, activation=hp_activation1)(x)
	hp_layer7=301
	x=tf.keras.layers.Dense(hp_layer7, activation=hp_activation1)(x)
	x=tf.keras.layers.Dense(1)(x)
	hp_learning_rate=0.0001
	model=tf.keras.Model(inputs=input_, outputs=x)
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),loss='MSE')
			
	return model

model=model_def()
model.load_weights('./cnn-model-best.hdf5')

model.summary()
pred=model.predict(test_ds)
print(r2_score(test_data['Heat Flux'],pred))
print(mean_absolute_percentage_error(test_data['Heat Flux'],pred))
plt.plot(test_data['Heat Flux'],pred[:,0],'o')
plt.plot([0,120],[0,120])
plt.show()
plt.plot(pred[:,0],'o')
plt.plot(test_data['Heat Flux'])
plt.show()


