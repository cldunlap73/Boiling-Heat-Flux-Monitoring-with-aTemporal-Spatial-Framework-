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

pcs=[1,10,20,30,40,100,200,300]
pcs=[5,15,25,35,50,60,150,250,400,600]
for i in range(len(pcs)):
	pc_num=pcs[i]
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
	def model_def():
		input_=tf.keras.layers.Input(shape=(pc_num))
		hp_layer1=1700
		hp_layer2=1000
		hp_layer3=100
		hp_layer4=800
		hp_layer5=600
		x=tf.keras.layers.Dense(hp_layer1, activation='relu')(input_)
		x=tf.keras.layers.Dense(hp_layer2, activation='relu')(x)
		x=tf.keras.layers.Dense(hp_layer3, activation='relu')(x)
		x=tf.keras.layers.Dense(hp_layer4, activation='relu')(x)
		x=tf.keras.layers.Dense(hp_layer5, activation='relu')(x)
		x=tf.keras.layers.Dense(1)(x)
		hp_learning_rate=0.0001
		model=tf.keras.Model(inputs=input_, outputs=x)
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),loss='MSE')
				
		return model



	checkpoint=ModelCheckpoint(f"./pca-mlp-pc{pc_num}-model-best.hdf5", save_best_only=True, save_weights_only=True, mode='min')

	stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
	model=model_def()

	history=model.fit(x_train,y_train, validation_data=(x_val,y_val),epochs=500,callbacks=[checkpoint,stop_early])
	train_loss=history.history['loss']
	val_loss=history.history['val_loss']
	loss_data=np.array([train_loss, val_loss])
	np.save(f'pca{pc_num}-mlp-static-loss.npy', loss_data)



