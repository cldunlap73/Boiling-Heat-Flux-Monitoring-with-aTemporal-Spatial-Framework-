# Code for plots of sequence length vs r2 for different downsampled data
# this will be for the pca-cnn models

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
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18,    # Default font size
                     'axes.titlesize': 20,   # Title font size
                     'axes.labelsize': 20,   # Axes label font size
                     'xtick.labelsize': 18,  # X-axis tick label font size
                     'ytick.labelsize': 18,
                     'legend.fontsize': 18}) # Y-axis tick label font size
# Define Test Function
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def train_model(pc_num, seq_len,ds_factor, file_path):
	x_test=np.load('./pca-test.npy')
	df_test=pd.read_csv('./Test_full_set.csv')
	y_test=np.array(df_test['Heat Flux'].tolist())
	x_test=x_test[:,:pc_num]
	
	
	# Added for downsampling
	def downsample(arr, factor):
		dsarray=arr[::factor]
		return dsarray
	
	x_test=downsample(x_test, ds_factor)
	y_test=downsample(y_test,ds_factor)
	
	
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

	
	
	x_test=np.reshape(x_test, (x_test.shape[0],x_test.shape[1],x_test.shape[2],1))

	
	inputs=tf.keras.Input(shape=(x_test.shape[1],x_test.shape[2],1))
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
			
	model.load_weights(file_path)

	predict=model.predict(x_test)
	
	return predict, y_test

# 
vals=[[5,10],[5,30],[5,40],[5,60],[5,100],[5,200],[5,400],
[10,10],[10,30],[10,40],[10,60],[10,100],[10,200],[10,400],
[40,10],[40,30],[40,40],[40,60],[40,100],[40,200]]

ds_vals=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
ds_fps=[50,38,30,25,22,19,17,15,14,13,12,11,10,9,9,9,8,8]
ds_vals=[3,4,5,6,7,8,9,10]
ds_fps=[50,38,30,25,22,19,17,15,14,13,12,11,10,9,9,9,8,8]
pcs=[5,10,40]
pcs=[40]
sls=[10,30,40,60,100,200]
ds_vals=[3,5,6,10]
fps=[150,75,50,30,25,15]
ds_fps=[50,30,25,15]
for i in range(len(pcs)):
	r2_vals=[]
	for k in range(len(sls)):
		file_path=f'./cnn-pc{pcs[i]}_sl{sls[k]}.hdf5'
		predict, y_test=train_model(pcs[i], sls[k],1, file_path)
		r2_vals.append(r2_score(y_test,predict))
	plt.plot(sls,r2_vals,linestyle='dashed', marker='s', markerfacecolor='none' ,label=f'{fps[0]} fps')
	r2_vals=[]
	for k in range(len(sls)):
		file_path=f"./ds_cnn-pc{pcs[i]}_sl{sls[k]}.hdf5"
		predict, y_test=train_model(pcs[i], sls[k],1, file_path)
		r2_vals.append(r2_score(y_test,predict))
	plt.plot(sls,r2_vals,linestyle='dashed', marker='s', markerfacecolor='none' ,label=f'{fps[1]} fps')
	for j in range(len(ds_vals)):
		r2_vals=[]

		for k in range(len(sls)):
			file_path=f"./ds{ds_vals[j]}_cnn-pc{pcs[i]}_sl{sls[k]}.hdf5"
			predict, y_test=train_model(pcs[i], sls[k],ds_vals[j], file_path)
			r2_vals.append(r2_score(y_test,predict))
		plt.plot(sls,r2_vals,linestyle='dashed', marker='s', markerfacecolor='none' ,label=f'{fps[j+2]} fps')
	plt.legend()
	plt.xlabel('Sequence Length (# of Images)')
	plt.ylabel(r'$R^2$')
	plt.savefig('./Figures/ds_sl_r2.png')      
	plt.show()
	plt.close()
	
for i in range(len(pcs)):
	r2_vals=[]
	temp=[]
	for k in range(len(sls)):
		file_path=f'./cnn-pc{pcs[i]}_sl{sls[k]}.hdf5'
		predict, y_test=train_model(pcs[i], sls[k],1, file_path)
		r2_vals.append(r2_score(y_test,predict))
		temp.append(sls[k]/150)
	plt.plot(temp,r2_vals, linestyle='dashed', marker='s', markerfacecolor='none',label=f'{fps[0]} fps')
	
	r2_vals=[]
	temp=[]
	for k in range(len(sls)):
		file_path=f"./ds_cnn-pc{pcs[i]}_sl{sls[k]}.hdf5"
		predict, y_test=train_model(pcs[i], sls[k],1, file_path)
		r2_vals.append(r2_score(y_test,predict))
		temp.append(sls[k]/75)
	plt.plot(temp,r2_vals, linestyle='dashed', marker='s', markerfacecolor='none',label=f'{fps[1]} fps')
	
	for j in range(len(ds_vals)):
		
		r2_vals=[]
		temp=[]
		for k in range(len(sls)):
			file_path=f"./ds{ds_vals[j]}_cnn-pc{pcs[i]}_sl{sls[k]}.hdf5"
			predict, y_test=train_model(pcs[i], sls[k],ds_vals[j],file_path)
			r2_vals.append(r2_score(y_test,predict))
			temp.append(sls[k]/ds_fps[j])
		plt.plot(temp,r2_vals, linestyle='dashed', marker='s', markerfacecolor='none',label=f'{fps[j+2]} fps')
	plt.xlabel('Temporal Sequence Length (s)')
	plt.ylabel(r'$R^2$')
	plt.legend()
	plt.savefig('./Figures/ds_tsl_r2.png')      
	plt.show()
	plt.close()


for i in range(len(pcs)):
	r2_vals=[]
	for k in range(len(sls)):
		file_path=f'./cnn-pc{pcs[i]}_sl{sls[k]}.hdf5'
		predict, y_test=train_model(pcs[i], sls[k],1, file_path)
		r2_vals.append(mean_absolute_percentage_error(y_test,predict))

	r2_vals=np.array(r2_vals)*100
	plt.plot(sls,r2_vals,linestyle='dashed', marker='s', markerfacecolor='none' ,label=f'{fps[0]} fps')
	r2_vals=[]
	for k in range(len(sls)):
		file_path=f"./ds_cnn-pc{pcs[i]}_sl{sls[k]}.hdf5"
		predict, y_test=train_model(pcs[i], sls[k],1, file_path)
		r2_vals.append(mean_absolute_percentage_error(y_test,predict))

	r2_vals=np.array(r2_vals)*100
	plt.plot(sls,r2_vals,linestyle='dashed', marker='s', markerfacecolor='none' ,label=f'{fps[1]} fps')
	for j in range(len(ds_vals)):
		r2_vals=[]

		for k in range(len(sls)):
			file_path=f"./ds{ds_vals[j]}_cnn-pc{pcs[i]}_sl{sls[k]}.hdf5"
			predict, y_test=train_model(pcs[i], sls[k],ds_vals[j], file_path)
			r2_vals.append(mean_absolute_percentage_error(y_test,predict))
		r2_vals=np.array(r2_vals)*100
		plt.plot(sls,r2_vals,linestyle='dashed', marker='s', markerfacecolor='none' ,label=f'{fps[j+2]} fps')
	plt.legend()
	plt.xlabel('Sequence Length (# of Images)')
	plt.ylabel(r'MAPE (%)')
	plt.savefig('./Figures/ds_sl_mape.png')      
	plt.show()
	plt.close()
	
for i in range(len(pcs)):
	r2_vals=[]
	temp=[]
	for k in range(len(sls)):
		file_path=f'./cnn-pc{pcs[i]}_sl{sls[k]}.hdf5'
		predict, y_test=train_model(pcs[i], sls[k],1, file_path)
		r2_vals.append(mean_absolute_percentage_error(y_test,predict))

		temp.append(sls[k]/150)
	r2_vals=np.array(r2_vals)*100
	plt.plot(temp,r2_vals, linestyle='dashed', marker='s', markerfacecolor='none',label=f'{fps[0]} fps')
	
	r2_vals=[]
	temp=[]
	for k in range(len(sls)):
		file_path=f"./ds_cnn-pc{pcs[i]}_sl{sls[k]}.hdf5"
		predict, y_test=train_model(pcs[i], sls[k],1, file_path)
		r2_vals.append(mean_absolute_percentage_error(y_test,predict))

		temp.append(sls[k]/75)
	r2_vals=np.array(r2_vals)*100
	plt.plot(temp,r2_vals, linestyle='dashed', marker='s', markerfacecolor='none',label=f'{fps[1]} fps')
	
	for j in range(len(ds_vals)):
		
		r2_vals=[]
		temp=[]
		for k in range(len(sls)):
			file_path=f"./ds{ds_vals[j]}_cnn-pc{pcs[i]}_sl{sls[k]}.hdf5"
			predict, y_test=train_model(pcs[i], sls[k],ds_vals[j],file_path)
			r2_vals.append(mean_absolute_percentage_error(y_test,predict))
		
			temp.append(sls[k]/ds_fps[j])
		r2_vals=np.array(r2_vals)*100	
		plt.plot(temp,r2_vals, linestyle='dashed', marker='s', markerfacecolor='none',label=f'{fps[j+2]} fps')
	plt.xlabel('Temporal Sequence Length (s)')
	plt.savefig('./Figures/ds_tsl_mape.png')      
	plt.show()
	plt.close()
