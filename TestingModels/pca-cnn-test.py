''' 

'''
import tensorflow as tf
import numpy as np
import random
import keras_tuner as kt
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd 
import json
import os 
from sklearn.metrics import r2_score, mean_absolute_percentage_error

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def train_model(pc_num, seq_len):
	x_test=np.load('./pca-test.npy')
	df_test=pd.read_csv('./Test_full_set.csv')
	y_test=np.array(df_test['Heat Flux'].tolist())
	x_test=x_test[:,:pc_num]
	def downsample(arr):
		dsarray=arr[::2]
		return dsarray
	x_test=downsample(x_test)
	y_test=downsample(y_test)
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
	
	
	
	amounttest=len(x_test)
	

	x_test=np.reshape(x_test, (x_test.shape[0],x_test.shape[1],x_test.shape[2],1))


	
	# define model
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
			
	
	model.load_weights(f"./ds_cnn-pc{pc_num}_sl{seq_len}.hdf5")
	model.summary()
	predict=model.predict(x_test)
	'''
	plt.plot(predict, y_test,'o')
	plt.plot([0,120],[0,120])
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.show()
	'''
	return predict, y_test
'''
vals=[[5,10],[5,30],[5,40],[5,60],[5,100],[5,200],[5,400],
[10,10],[10,30],[10,40],[10,60],[10,100],[10,200],[10,400],
[40,10],[40,30],[40,40],[40,60],[40,100],[40,200]]
import matplotlib.pyplot as plt
for i in range(len(vals)):	
	print('Number of Pcs', vals[i][0], 'seq len', vals[i][1])
	pred, ytest=train_model(vals[i][0],vals[i][1])
	plt.plot(ytest,pred,'o')
	plt.plot([1,120],[1,120])	
	plt.show()
	print(vals[i])
	print(mean_absolute_percentage_error(ytest,pred))
	print(r2_score(ytest,pred))
	plt.plot(pred,'o')
	plt.plot(ytest)


	plt.show()
'''
import matplotlib.pyplot as plt
sets=[[5,10],[5,30],[5,40],[5,60],[5,100],[5,200],
[9,10],[9,30],[9,40],[9,60],[9,100],[9,200],
[10,10],[10,30],[10,40],[10,60],[10,100],[10,200],
[20,10],[20,30],[20,40],[20,60],[20,100],[20,200],
[30,10],[30,30],[30,40],[30,60],[30,100],[30,200],
[40,10],[40,30],[40,40],[40,60],[40,100],[40,200]]
# Create a 6x2 grid of subplots

sets=[[5,10],[5,30],[5,40],[5,60],[5,100],[5,200],[5,400],
[10,10],[10,30],[10,40],[10,60],[10,100],[10,200],[10,400],
[40,10],[40,30],[40,40],[40,60],[40,100],[40,200]]
sets =[[40,200]]
fig, axes = plt.subplots(3, 7, figsize=(12, 4))

# Flatten the 2D array of subplots into a 1D array

axes = axes.flatten()

# Plot your figures
for i in range(len(sets)):
    
    ax = axes[i]
    pred,ytest=train_model(sets[i][0],sets[i][1])
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

fig, axes = plt.subplots(3, 7, figsize=(12, 4))

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

# Show the figure
plt.show()
	
