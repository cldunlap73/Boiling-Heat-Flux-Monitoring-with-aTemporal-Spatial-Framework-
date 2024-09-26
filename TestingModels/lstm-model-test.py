# code for determining hyper parameters for pca-lstm model
# import libraries

'''
This code implements a lstm model 
'''
# Import Libraries
import tensorflow as tf
import numpy as np
import random
import keras_tuner as kt
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd 
import json
import os
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# Load data
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def train_model(pc_num, seq_len):
	x_test=np.load('./pca-test.npy')
	df_test=pd.read_csv('./Test_full_set.csv')
	y_test=np.array(df_test['Heat Flux'].tolist())
	x_test=x_test[:,:pc_num]
	
	
	def split_data(signal,sl,st):
    		length=len(signal)
    		amt_samples=int((length-sl)/st)
    		data=np.empty((amt_samples,sl))
    		for i in range(amt_samples):
        		data[i]=signal[(i*st):(i*st)+sl]
        
    		return data

	index=[i for i in range(len(x_test))]
	index=split_data(index, seq_len, 1).astype(int)
	x_test=x_test[index]
	y_test=split_data(y_test,seq_len,1)[:,-1]
	print(x_test.shape, y_test.shape)
	condition=y_test>15
	x_test=x_test[condition]
	y_test=y_test[condition]
	
	# define model
	def model_def():
		input_=tf.keras.layers.Input(shape=(seq_len, pc_num))
		hp_layer1=590
		x=tf.keras.layers.LSTM(hp_layer1,  return_sequences=True)(input_)
		x=tf.keras.layers.Dropout(0.2)(x)
		hp_layer2=990
		x=tf.keras.layers.LSTM(hp_layer2,  return_sequences=True)(x)
		hp_layer3=490
		x=tf.keras.layers.LSTM(hp_layer3,  return_sequences=True)(x)
		x=tf.keras.layers.Dropout(0.2)(x)
		hp_layer4=930
		x=tf.keras.layers.LSTM(hp_layer4,  return_sequences=True)(x)
		x=tf.keras.layers.Dropout(0.2)(x)
		hp_layer5=190
		x=tf.keras.layers.LSTM(hp_layer5,  return_sequences=True)(x)
		x=tf.keras.layers.Dropout(0.2)(x)
		hp_layer6=560
		x=tf.keras.layers.LSTM(hp_layer6)(x)
		hp_layer7=260
		x=tf.keras.layers.Dense(hp_layer7, activation='relu')(x)
		hp_layer8=440
		x=tf.keras.layers.Dense(hp_layer8, activation='relu')(x)
		hp_layer9=120
		x=tf.keras.layers.Dense(hp_layer9, activation='relu')(x)
		hp_layer10=870
		x=tf.keras.layers.Dense(hp_layer10, activation='relu')(x)
		x=tf.keras.layers.Dense(1)(x)
		model=tf.keras.models.Model(inputs=input_,outputs=x)
		hp_lr=0.001
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),loss='MSE')
		return model
		
	model=model_def()

	

	model=model_def()
	model.load_weights(f"./lstm_pc{pc_num}_sl{seq_len}-model-best.hdf5")
	model.summary()
	predict=model.predict(x_test)
	
	return predict, y_test

	
import matplotlib.pyplot as plt

sets=[[5,2],[5,10],[5,30],[5,40],[5,60],[5,100],[5,200],[5,400],
[10,2],[10,10],[10,30],[10,40],[10,60],[10,100],[10,200],[10,400],
[40,2],[40,10],[40,30],[40,40],[40,60],[40,100],[40,200]]
'''
vals=[[7,10],[7,30],[7,40],[7,60],[7,100],[7,200],[7,400],
[15,10],[15,30],[15,40],[15,60],[15,100],[15,200],[15,400]] 
vals=[[9,5],[9,10],[9,15],[9,30],[9,40],[9,60],[9,100],[9,200],[9,400],[9,800],
[20,5],[20,10],[20,15],[20,30],[20,40],[20,60],[20,100],[20,200],[20,400],[20,800],
[30,5],[30,10],[30,15],[30,30],[30,40],[30,60],[30,100],[30,200],[30,400],[30,800]] 
'''
sets=[[40,200]]
fig, axes = plt.subplots(3, 6, figsize=(12, 4))

# Flatten the 2D array of subplots into a 1D array

axes = axes.flatten()

# Plot your figures
for i in range(len(sets)):
    
    ax = axes[i]
    pred,ytest=train_model(sets[i][0],sets[i][1])
    print(pred.shape, ytest.shape)
    print(r2_score(ytest,pred))
    # Example plot
    ax.plot(pred,'o')
    ax.plot(ytest)
    string="{:.2f}".format(r2_score(ytest,pred))+","+"{:.2f}".format(mean_absolute_percentage_error(ytest,pred))
    ax.text(0,85,string)
    
    ax.set_title(f'pc{sets[i][0]} sl{sets[i][1]}')  # Set subplot title

# Adjust spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()

fig, axes = plt.subplots(3, 6, figsize=(12, 4))

# Flatten the 2D array of subplots into a 1D array

axes = axes.flatten()

# Plot your figures
for i in range(len(sets)):
    
    ax = axes[i]
    pred,ytest=train_model(sets[i][0],sets[i][1])
    print(r2_score(ytest,pred))
    # Example plot
    ax.plot(pred,ytest,'o')
    ax.plot([1,100],[1,100])
    string="{:.3f}".format(r2_score(ytest,pred))+","+"{:.3f}".format(mean_absolute_percentage_error(ytest,pred))
    ax.text(0,85,string)
    ax.set_title(f'pc{sets[i][0]} sl{sets[i][1]}')  # Set subplot title

# Adjust spacing between subplots
plt.tight_layout()
		
