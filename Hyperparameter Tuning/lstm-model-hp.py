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
import pandas as pd 
import json
import os

# Load data
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
	def model_def(hp):
		input_=tf.keras.layers.Input(shape=(seq_len, pc_num))
		hp_layer1=hp.Int('layer1', min_value=10, max_value=1000, step=10)
		x=tf.keras.layers.LSTM(hp_layer1,  return_sequences=True)(input_)
		x=tf.keras.layers.Dropout(0.2)(x)
		hp_layer2=hp.Int('layer2', min_value=10, max_value=1000, step=10)
		x=tf.keras.layers.LSTM(hp_layer2,  return_sequences=True)(x)
		hp_layer3=hp.Int('layer3', min_value=10, max_value=1000, step=10)
		x=tf.keras.layers.LSTM(hp_layer3,  return_sequences=True)(x)
		x=tf.keras.layers.Dropout(0.2)(x)
		hp_layer4=hp.Int('layer4', min_value=10, max_value=1000, step=10)
		x=tf.keras.layers.LSTM(hp_layer4,  return_sequences=True)(x)
		x=tf.keras.layers.Dropout(0.2)(x)
		hp_layer5=hp.Int('layer5', min_value=10, max_value=1000, step=10)
		x=tf.keras.layers.LSTM(hp_layer5,  return_sequences=True)(x)
		x=tf.keras.layers.Dropout(0.2)(x)
		hp_layer6=hp.Int('layer6', min_value=10, max_value=1000, step=10)
		x=tf.keras.layers.LSTM(hp_layer6,  return_sequences=True)(x)
		hp_layer7=hp.Int('layer7', min_value=10, max_value=1000, step=10)
		x=tf.keras.layers.Dense(hp_layer7, activation='relu')(x)
		hp_layer8=hp.Int('layer8', min_value=10, max_value=1000, step=10)
		x=tf.keras.layers.Dense(hp_layer8, activation='relu')(x)
		hp_layer9=hp.Int('layer9', min_value=10, max_value=1000, step=10)
		x=tf.keras.layers.Dense(hp_layer9, activation='relu')(x)
		hp_layer10=hp.Int('layer10', min_value=10, max_value=1000, step=10)
		x=tf.keras.layers.Dense(hp_layer10, activation='relu')(x)
		x=tf.keras.layers.Dense(1)(x)
		model=tf.keras.models.Model(inputs=input_,outputs=x)
		hp_lr=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),loss='MSE')
		return model
		
		
	tuner = kt.Hyperband(model_def,
                     	objective='val_loss',
                     	max_epochs=100,
                     	factor=3,
                     	directory='./Tuning-Models/PCA/LSTM',
                     	project_name=f'prim-lstm-pcs{pc_num}_sl{seq_len}')
                     

	stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

	tuner.search(x_train, y_train, validation_data=(x_val,y_val),epochs=100,callbacks=[stop_early])
	
	

	# Assuming you have performed tuner.search() and have a tuner object named 'tuner'
	best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
	best_hyperparameters_dict = best_hyperparameters.get_config()  # Get best hyperparameters as a dictionary

	output_file_path = f'./Tuning-Models/PCA/LSTM/lstm-pcs{pc_num}_sl{seq_len}/best-hyperparameters.txt'
	with open(output_file_path, 'w') as f:
    		f.write(json.dumps(best_hyperparameters_dict, indent=4))  # Write best hyperparameters as JSON string

	print(f"Best hyperparameters saved to '{output_file_path}'.")
	
pc_nums=[40]
seq_lens=[200]
for pc_num in pc_nums:
	for seq_len in seq_lens:
		gen_model(pc_num,seq_len)
		
