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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pc_num=40
seq_len=200
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
train_amt=int(1*amounttrain)
shuffle_train=shuffle_idx[:train_amt]
np.save(f'./norm_train_pca{pc_num}_sl{seq_len}_random_shuffle.npy',np.array(shuffle_train))

amountval=len(x_val)
indexs=[i for i in range(amountval)]
random.shuffle(indexs)
shuffle_idx=indexs
val_amt=int(1*amountval)
shuffle_val=shuffle_idx[:val_amt]
np.save(f'./norm_val_pca{pc_num}_sl{seq_len}_random_shuffle.npy',np.array(shuffle_val))

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
	
def model_def(hp):
	inputs=tf.keras.Input(shape=(x_train.shape[1],x_train.shape[2],1))
	hp_layer1_filter=hp.Int('layer1_filter', min_value=32, max_value=100, step=10)
	hp_layer1_kernel=hp.Int('layer1_kernel', min_value=3, max_value=5, step=2)
	hp_layer1_stride=hp.Int('layer1_stride', min_value=1, max_value=3, step=2)
	x=tf.keras.layers.Conv2D(hp_layer1_filter, hp_layer1_kernel,strides=hp_layer1_stride, activation='relu')(inputs)
	#x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
	x=tf.keras.layers.Dropout(0.2)(x)
	hp_layer2_filter=hp.Int('layer2_filter', min_value=32, max_value=100, step=10)
	hp_layer2_kernel=hp.Int('layer2_kernel', min_value=3, max_value=5, step=2)
	hp_layer2_stride=hp.Int('layer2_stride', min_value=1, max_value=3, step=2)
	x = tf.keras.layers.Conv2D(filters=hp_layer2_filter, kernel_size=hp_layer2_kernel,strides= hp_layer2_stride,padding='same', activation='relu')(x)
	hp_layer3_filter=hp.Int('layer3_filter', min_value=32, max_value=100, step=10)
	hp_layer3_kernel=hp.Int('layer3_kernel', min_value=3, max_value=5, step=2)
	hp_layer3_stride=hp.Int('layer3_stride', min_value=1, max_value=3, step=2)
	x = tf.keras.layers.Conv2D(filters=hp_layer3_filter, kernel_size=hp_layer3_kernel, strides=hp_layer3_stride, padding='same',activation='relu')(x)
	x = tf.keras.layers.MaxPooling2D(pool_size=2,padding='same')(x)
	hp_layer4_filter=hp.Int('layer4_filter', min_value=32, max_value=100, step=10)
	hp_layer4_kernel=hp.Int('layer4_kernel', min_value=3, max_value=5, step=2)
	hp_layer4_stride=hp.Int('layer4_stride', min_value=1, max_value=3, step=2)
	x = tf.keras.layers.Conv2D(filters=hp_layer4_filter, kernel_size=hp_layer4_kernel, strides=hp_layer4_stride,padding='same', activation='relu')(x)
	x=tf.keras.layers.Dropout(0.2)(x)
	#x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
	x = tf.keras.layers.Flatten()(x)
	hp_layer5_neurons=hp.Int('layer5_neurons', min_value=1, max_value=1000, step=10)
	x = tf.keras.layers.Dense(units=hp_layer5_neurons, activation='relu')(x)
	hp_layer6_neurons=hp.Int('layer6_neurons', min_value=1, max_value=1000, step=10)
	x = tf.keras.layers.Dense(units=hp_layer6_neurons, activation='relu')(x)
	outputs = tf.keras.layers.Dense(units=1)(x)

	model=tf.keras.models.Model(inputs=inputs, outputs=outputs)
	
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='MSE')
			
	return model
	

tuner=kt.Hyperband(model_def,
		objective='val_loss',
		max_epochs=100,
		factor=3,
		directory='./Tuning-Models/PCA',
		project_name='cnn')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

tuner.search(x_train,y_train, validation_data=(x_val,y_val),epochs=50,callbacks=[stop_early])

# Assuming you have performed tuner.search() and have a tuner object named 'tuner'
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
best_hyperparameters_dict = best_hyperparameters.get_config()  # Get best hyperparameters as a dictionary

output_file_path = f'./Tuning-Models/cnn_full/best-hyperparameters.txt'
with open(output_file_path, 'w') as f:
    	f.write(json.dumps(best_hyperparameters_dict, indent=4))  
    	# Write best hyperparameters as JSON string
    	
    	

	
