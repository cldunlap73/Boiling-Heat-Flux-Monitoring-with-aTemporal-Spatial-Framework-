''' 
Code for training lstm model
'''
import tensorflow as tf
import numpy as np
import random
import keras_tuner as kt
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd 
import json
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def train_model(pc_num, seq_len,ds_factor):
	x_train=np.load('./pca-train.npy')
	df_train=pd.read_csv('./Train_full_set.csv')
	y_train=np.array(df_train['Heat Flux'].tolist())
	x_train=x_train[:,:pc_num]
	
	x_val=np.load('./pca-val.npy')
	df_val=pd.read_csv('./Val_full_set.csv')
	y_val=np.array(df_val['Heat Flux'].tolist())
	x_val=x_val[:,:pc_num]
	
	# Added for downsampling
	def downsample(arr, factor):
		dsarray=arr[::factor]
		return dsarray
	
	x_train=downsample(x_train, ds_factor)
	y_train=downsample(y_train,ds_factor)
	x_val=downsample(x_val,ds_factor)
	y_val=downsample(y_val,ds_factor)
	
	
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
	train_amt=int(1*amounttrain)
	shuffle_train=shuffle_idx[:train_amt]
	np.save(f'./ds{ds_factor}_norm_train_pca{pc_num}_sl{seq_len}_random_shuffle.npy',np.array(shuffle_train))
	
	amountval=len(x_val)
	indexs=[i for i in range(amountval)]
	random.shuffle(indexs)
	shuffle_idx=indexs
	val_amt=int(1*amountval)
	shuffle_val=shuffle_idx[:val_amt]
	np.save(f'./ds{ds_factor}_norm_val_pca{pc_num}_sl{seq_len}_random_shuffle.npy',np.array(shuffle_val))

	x_train=x_train[shuffle_train]
	y_train=y_train[shuffle_train]
	
	x_val=x_val[shuffle_val]
	y_val=y_val[shuffle_val]
	print(np.amax(x_train))
	print(x_train.shape,y_train.shape, x_val.shape, y_val.shape)
	
	
	x_train=np.reshape(x_train, (x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
	x_val=np.reshape(x_val, (x_val.shape[0],x_val.shape[1],x_val.shape[2],1))
	print(max(y_train))
	print(x_train.shape,y_train.shape, x_val.shape, y_val.shape)
	# filter out lower heat flux data
	condition=y_train>15
	x_train=x_train[condition]
	y_train=y_train[condition]
	condition=y_val>15
	x_val=x_val[condition]
	y_val=y_val[condition]
	#x_train=x_train[:-20000]
	#y_train=y_train[:-20000]
	# define model
	inputs=tf.keras.Input(shape=(x_train.shape[1],x_train.shape[2],1))
	hp_layer1_filter=72
	hp_layer1_kernel=3
	hp_layer1_stride=3
	x=tf.keras.layers.Conv2D(hp_layer1_filter, hp_layer1_kernel,strides=hp_layer1_stride, activation='relu')(inputs)
	#x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
	x=tf.keras.layers.Dropout(0.2)(x)
	hp_layer2_filter=72
	hp_layer2_kernel=5
	hp_layer2_stride=1
	x = tf.keras.layers.Conv2D(filters=hp_layer2_filter, kernel_size=hp_layer2_kernel,strides= hp_layer2_stride,padding='same', activation='relu')(x)
	hp_layer3_filter=92
	hp_layer3_kernel=5
	hp_layer3_stride=3
	x = tf.keras.layers.Conv2D(filters=hp_layer3_filter, kernel_size=hp_layer3_kernel, strides=hp_layer3_stride, padding='same',activation='relu')(x)
	x = tf.keras.layers.MaxPooling2D(pool_size=2,padding='same')(x)
	hp_layer4_filter=62
	hp_layer4_kernel=3
	hp_layer4_stride=1
	x = tf.keras.layers.Conv2D(filters=hp_layer4_filter, kernel_size=hp_layer4_kernel, strides=hp_layer4_stride,padding='same', activation='relu')(x)
	x=tf.keras.layers.Dropout(0.2)(x)
	#x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
	x = tf.keras.layers.Flatten()(x)
	hp_layer5_neurons=181
	x = tf.keras.layers.Dense(units=hp_layer5_neurons, activation='relu')(x)
	hp_layer6_neurons=931
	x = tf.keras.layers.Dense(units=hp_layer6_neurons, activation='relu')(x)
	outputs = tf.keras.layers.Dense(units=1)(x)

	model=tf.keras.models.Model(inputs=inputs, outputs=outputs)
	
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='MSE')
			
	checkpoint=ModelCheckpoint(f"./ds{ds_factor}_cnn-pc{pc_num}_sl{seq_len}.hdf5", save_best_only=True, save_weights_only=True, mode='min')
	early_stop=EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

	history=model.fit(x_train,y_train, validation_data=(x_val,y_val), epochs=600, callbacks=[checkpoint, early_stop])
	
	train_loss=history.history['loss']
	val_loss=history.history['val_loss']
	loss_data=np.array([train_loss, val_loss])
	np.save(f'ds{ds_factor}_loss_pc{pc_num}_sl{seq_len}.npy',loss_data)

vals=[[5,10],[5,30],[5,40],[5,60],[5,100],[5,200],[5,400],
[10,10],[10,30],[10,40],[10,60],[10,100],[10,200],[10,400],
[40,10],[40,30],[40,40],[40,60],[40,100],[40,200]]
'''
vals=[[7,10],[7,30],[7,40],[7,60],[7,100],[7,200],[7,400],
[15,10],[15,30],[15,40],[15,60],[15,100],[15,200],[15,400]] 

vals=[[9,5],[9,10],[9,15],[9,30],[9,40],[9,60],[9,100],[9,200],[9,400],[9,800],
[20,5],[20,10],[20,15],[20,30],[20,40],[20,60],[20,100],[20,200],[20,400],[20,800],
[30,5],[30,10],[30,15],[30,30],[30,40],[30,60],[30,100],[30,200],[30,400],[30,800]]
 
ds_vals=[3,5,6,7,8]
'''
ds_vals=[15,16,17,18,19,20]
for j in range(len(ds_vals)):
	for i in range(len(vals)):
		train_model(vals[i][0],vals[i][1],ds_vals[j])

	
