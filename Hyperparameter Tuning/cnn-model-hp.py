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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Create Dataset
csv_train_loc="./Train_full_set.csv"
csv_val_loc="./Val_full_set.csv"
img_w,img_h=200,200
dftrain=pd.read_csv(csv_train_loc)
dftrain=dftrain.to_numpy()
amounttrain=len(dftrain)
indexs=[i for i in range(amounttrain)]
random.shuffle(indexs)
shuffle_idx=indexs
train_amt=int(0.2*amounttrain)
shuffle_train=shuffle_idx[:train_amt]
np.save('./tuning_train_random_shuffle.npy',np.array(shuffle_train))

dfval=pd.read_csv(csv_val_loc)
dfval=dfval.to_numpy()
amountval=len(dfval)
indexs=[i for i in range(amountval)]
random.shuffle(indexs)
shuffle_idx=indexs
val_amt=int(0.1*amountval)
shuffle_val=shuffle_idx[:val_amt]
np.save('./tuning_val_random_shuffle.npy',np.array(shuffle_val))


train_data={'Image': (np.array(dftrain[:,3]).astype('U')[shuffle_train]),
            'Heat Flux': (np.array(dftrain[:,1]).astype('float')[shuffle_train])}

val_data={'Image': (np.array(dfval[:,3]).astype('U')[shuffle_val]),
          'Heat Flux': (np.array(dfval[:,1]).astype('float')[shuffle_val])} 

    
@tf.function
def load_and_preprocess_image(path):
    image_string = tf.io.read_file(path)
    img = tf.io.decode_png(image_string, channels=1)
    img = tf.image.resize(img, [img_w, img_h])
    img = tf.cast(img, tf.float32) / 255.0 
    return img

def load_and_preprocess_from_path_labels(image_path, labels):
    image = load_and_preprocess_image(image_path)
    return image, labels

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=300)
    ds = ds.batch(50)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

train_ds = tf.data.Dataset.from_generator(lambda: zip(train_data['Image'], train_data['Heat Flux']), (tf.string, tf.float32))
train_ds = train_ds.map(load_and_preprocess_from_path_labels, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = configure_for_performance(train_ds)

val_ds = tf.data.Dataset.from_generator(lambda: zip(val_data['Image'], val_data['Heat Flux']), (tf.string, tf.float32))
val_ds = val_ds.map(load_and_preprocess_from_path_labels, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = configure_for_performance(val_ds)

for s1,s3 in train_ds.take(1):
    print("Image shape: ", s1.numpy().shape)
    print("Label: ", s3.numpy().shape)


# Define Model
def model_def(hp):
	input_=tf.keras.layers.Input(shape=(200,200,1))
	hp_activation=hp.Choice('activation', values=['relu','tanh'])
	hp_layer1_filter=hp.Int('layer1_filter', min_value=32, max_value=400, step=10)
	hp_layer1_kernel=hp.Int('layer1_kernel', min_value=3, max_value=5, step=2)
	hp_layer1_stride=hp.Int('layer1_stride', min_value=1, max_value=3, step=2)
	x=tf.keras.layers.Conv2D(hp_layer1_filter, hp_layer1_kernel,hp_layer1_stride, activation=hp_activation)(input_)
	hp_layer2_kernel=hp.Int('layer2_kernel', min_value=2, max_value=6, step=2)
	hp_layer2_stride=hp.Int('layer2_stride', min_value=1, max_value=4, step=2)
	x=tf.keras.layers.MaxPool2D(hp_layer2_kernel,hp_layer2_stride)(x)
	x=tf.keras.layers.Dropout(0.2)(x)
	hp_layer3_filter=hp.Int('layer3_filter', min_value=32, max_value=500, step=10)
	hp_layer3_kernel=hp.Int('layer3_kernel', min_value=3, max_value=5, step=2)
	hp_layer3_stride=hp.Int('layer3_stride', min_value=1, max_value=3, step=2)
	x=tf.keras.layers.Conv2D(hp_layer3_filter, hp_layer3_kernel,hp_layer3_stride, activation=hp_activation)(x)
	hp_layer4_kernel=hp.Int('layer4_kernel', min_value=2, max_value=6, step=2)
	hp_layer4_stride=hp.Int('layer4_stride', min_value=1, max_value=4, step=2)
	x=tf.keras.layers.MaxPool2D(hp_layer4_kernel, hp_layer4_stride)(x)
	hp_layer5_filter=hp.Int('layer5_filter', min_value=32, max_value=300, step=10)
	hp_layer5_kernel=hp.Int('layer5_kernel', min_value=3, max_value=5, step=2)
	hp_layer5_stride=hp.Int('layer5_stride', min_value=1, max_value=4, step=2)
	x=tf.keras.layers.Conv2D(hp_layer5_filter, hp_layer5_kernel,hp_layer5_stride, activation=hp_activation)(x)
	x=tf.keras.layers.GlobalAveragePooling2D()(x)
	x=tf.keras.layers.Dropout(0.2)(x)
	hp_activation1=hp.Choice('activation', values=['relu','tanh'])
	hp_layer6=hp.Int('layer6', min_value=1, max_value=1000, step=100)
	x=tf.keras.layers.Dense(hp_layer6, activation=hp_activation1)(x)
	hp_layer7=hp.Int('layer7', min_value=1, max_value=1000, step=100)
	x=tf.keras.layers.Dense(hp_layer7, activation=hp_activation1)(x)
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
                     project_name='cnn_full')
# Training Loop

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

tuner.search(train_ds, validation_data=val_ds,epochs=50,callbacks=[stop_early])

# Assuming you have performed tuner.search() and have a tuner object named 'tuner'
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
best_hyperparameters_dict = best_hyperparameters.get_config()  # Get best hyperparameters as a dictionary

output_file_path = f'./Tuning-Models/cnn_full/best-hyperparameters.txt'
with open(output_file_path, 'w') as f:
    	f.write(json.dumps(best_hyperparameters_dict, indent=4))  # Write best hyperparameters as JSON string

print(f"Best hyperparameters saved to '{output_file_path}'.")

