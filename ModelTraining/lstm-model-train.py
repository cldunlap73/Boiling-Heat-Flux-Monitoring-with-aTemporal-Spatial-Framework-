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

# Load data
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def gen_model(pc_num, seq_len):
	x_train=np.load('./pca-train.npy')
	df_train=pd.read_csv('./Train_full_set.csv')
	y_train=np.array(df_train['Heat Flux'].tolist())
	x_train=x_train[:,:pc_num]
	
	x_val=np.load('./pca-val.npy')
	df_val=pd.read_csv('./Val_full_set.csv')
	y_val=np.array(df_val['Heat Flux'].tolist())
	x_val=x_val[:,:pc_num]
	
	def split_data(signal,sl,st):
    		length=len(signal)
    		amt_samples=int((length-sl)/st)
    		data=np.empty((amt_samples,sl))
    		for i in range(amt_samples):
        		data[i]=signal[(i*st):(i*st)+sl]
        
    		return data

	index=[i for i in range(len(x_train))]
	index=split_data(index, seq_len, 1).astype(int)
	x_train=x_train[index]
	y_train=split_data(y_train,seq_len,1)[:,-1]
	print(x_train.shape, y_train.shape)
	condition=y_train>15
	x_train=x_train[condition]
	y_train=y_train[condition]
	condition=y_val>15
	x_val=x_val[condition]
	y_val=y_val[condition]
	
	index=[i for i in range(len(x_val))]
	index=split_data(index, seq_len, 1).astype(int)
	x_val=x_val[index]
	y_val=split_data(y_val,seq_len,1)[:,-1]
	
	amounttrain=len(x_train)
	indexs=[i for i in range(amounttrain)]
	random.shuffle(indexs)
	shuffle_idx=indexs
	train_amt=int(0.1*amounttrain)
	shuffle_train=shuffle_idx[:train_amt]
	np.save(f'./prim-tuning_train_pca{pc_num}_sl{seq_len}_random_shuffle.npy',np.array(shuffle_train))
	#shuffle_train=np.load(f'./tuning_train_pca{pc_num}_sl{seq_len}_random_shuffle.npy')
	amountval=len(x_val)
	indexs=[i for i in range(amountval)]
	random.shuffle(indexs)
	shuffle_idx=indexs
	val_amt=int(0.1*amountval)
	shuffle_val=shuffle_idx[:val_amt]
	np.save(f'./prim-tuning_val_pca{pc_num}_sl{seq_len}_random_shuffle.npy',np.array(shuffle_val))
	#shuffle_val=np.load(f'./tuning_val_pca{pc_num}_sl{seq_len}_random_shuffle.npy')
	x_train=x_train[shuffle_train]
	y_train=y_train[shuffle_train]
	
	x_val=x_val[shuffle_val]
	y_val=y_val[shuffle_val]
	
	print(x_train.shape,y_train.shape, x_val.shape, y_val.shape)
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
	checkpoint=ModelCheckpoint(f"./lstm_pc{pc_num}_sl{seq_len}-model-best.hdf5", save_best_only=True, save_weights_only=True, mode='min')
	
	stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
	model=model_def()
	
	history=model.fit(x_train,y_train, validation_data=(x_val,y_val),epochs=400,callbacks=[checkpoint,stop_early])
	train_loss=history.history['loss']
	val_loss=history.history['val_loss']
	loss_data=np.array([train_loss, val_loss])
	np.save(f'loss_pc{pc_num}_sl{seq_len}_lstm.npy', loss_data)

	
pc_nums=[40]
seq_lens=[200]

vals=[[5,2],[5,10],[5,30],[5,40],[5,60],[5,100],[5,200],[5,400],
[10,2],[10,10],[10,30],[10,40],[10,60],[10,100],[10,200],[10,400],
[40,2],[40,10],[40,30],[40,40],[40,60],[40,100],[40,200]]
'''
vals=[[7,10],[7,30],[7,40],[7,60],[7,100],[7,200],[7,400],
[15,10],[15,30],[15,40],[15,60],[15,100],[15,200],[15,400]] 
vals=[[9,5],[9,10],[9,15],[9,30],[9,40],[9,60],[9,100],[9,200],[9,400],[9,800],
[20,5],[20,10],[20,15],[20,30],[20,40],[20,60],[20,100],[20,200],[20,400],[20,800],
[30,5],[30,10],[30,15],[30,30],[30,40],[30,60],[30,100],[30,200],[30,400],[30,800]] 
'''
for i in range(len(vals)):
	gen_model(vals[i][0],vals[i][1])
		
