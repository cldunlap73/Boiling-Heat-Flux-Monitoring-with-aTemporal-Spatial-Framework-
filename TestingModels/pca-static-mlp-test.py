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
from sklearn.metrics import r2_score, mean_absolute_percentage_error

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pcs=[1,5,10,15,20,25,30,35,40,50,100,150,200,250,300,400,600,900]
r2_scores=[]
mape_vals=[]
pcs=[900]
for i in range(len(pcs)):
	pc_num=pcs[i]
	x_test=np.load('./pca-test.npy')
	df_test=pd.read_csv('./Test_full_set.csv')
	y_test=np.array(df_test['Heat Flux'].tolist())
	x_test=x_test[:,:pc_num]


	# filter out lower heat flux data
	condition=y_test>15
	x_test=x_test[condition]
	y_test=y_test[condition]


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
	model.load_weights(f"./pca-mlp-pc{pc_num}-model-best.hdf5")
	model.summary()
	pred=model.predict(x_test)
	import matplotlib.pyplot as plt
	r2_scores.append(r2_score(y_test,pred))
	mape_vals.append(mean_absolute_percentage_error(y_test,pred))
	print(r2_score(y_test,pred))
	print(mean_absolute_percentage_error(y_test,pred))
	
plt.plot(pcs, r2_scores)
plt.show()
plt.plot(pcs,mape_vals)
plt.show()


