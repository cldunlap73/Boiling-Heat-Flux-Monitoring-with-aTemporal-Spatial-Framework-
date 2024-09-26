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
from sklearn.metrics import r2_score
import random
import os
import importlib
import keras_tuner as kt
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pc_num=900
x_train=np.load('./pca-train.npy')
df_train=pd.read_csv('./Train_full_set.csv')
y_train=np.array(df_train['Heat Flux'].tolist())
x_train=x_train[:,:pc_num]

x_val=np.load('./pca-val.npy')
df_val=pd.read_csv('./Val_full_set.csv')
y_val=np.array(df_val['Heat Flux'].tolist())
x_val=x_val[:,:pc_num]


print(x_train.shape,y_train.shape, x_val.shape, y_val.shape)


# filter out lower heat flux data
condition=y_train>15
x_train=x_train[condition]
y_train=y_train[condition]
condition=y_val>15
x_val=x_val[condition]
y_val=y_val[condition]


# Define Model
def model_def(hp):
	input_=tf.keras.layers.Input(shape=(900))
	hp_layer1=hp.Int('layer1_filter', min_value=100, max_value=2000, step=100)
	hp_layer2=hp.Int('layer2_filter', min_value=100, max_value=2000, step=100)
	hp_layer3=hp.Int('layer3_filter', min_value=100, max_value=2000, step=100)
	hp_layer4=hp.Int('layer4_filter', min_value=100, max_value=2000, step=100)
	hp_layer5=hp.Int('layer5_filter', min_value=100, max_value=2000, step=100)
	x=tf.keras.layers.Dense(hp_layer1, activation='relu')(input_)
	x=tf.keras.layers.Dense(hp_layer2, activation='relu')(x)
	x=tf.keras.layers.Dense(hp_layer3, activation='relu')(x)
	x=tf.keras.layers.Dense(hp_layer4, activation='relu')(x)
	x=tf.keras.layers.Dense(hp_layer5, activation='relu')(x)
	x=tf.keras.layers.Dense(1)(x)
	hp_learning_rate=hp.Choice('learning_rate', values=[1e-2,1e-3,1e-4])
	model=tf.keras.Model(inputs=input_, outputs=x)
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),loss='MSE')
			
	return model
strategy=tf.distribute.MirroredStrategy(devices=['GPU:0','GPU:1'])
with strategy.scope():
	tuner = kt.Hyperband(model_def,
                     objective='val_loss',
                     max_epochs=100,
                     factor=3,
                     directory='./Tuning-Models',
                     project_name='pca_static_mlp')
# Training Loop

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

tuner.search(x_train,y_train, validation_data=(x_val,y_val),epochs=50,callbacks=[stop_early])

# Assuming you have performed tuner.search() and have a tuner object named 'tuner'
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
best_hyperparameters_dict = best_hyperparameters.get_config()  # Get best hyperparameters as a dictionary

output_file_path = f'./Tuning-Models/cnn_full/best-hyperparameters.txt'
with open(output_file_path, 'w') as f:
    	f.write(json.dumps(best_hyperparameters_dict, indent=4))  # Write best hyperparameters as JSON string

print(f"Best hyperparameters saved to '{output_file_path}'.")

