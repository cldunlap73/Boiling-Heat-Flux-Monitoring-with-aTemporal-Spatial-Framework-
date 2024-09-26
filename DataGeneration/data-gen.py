# -*- coding: utf-8 -*-
"""
Code for generating csv file datasets
These sets include interpolated heat flux,time, and image paths.
"""
# load libraries
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def data_gen(image_loc,image_start_sec,temp_loc,fps,sl,st):
	image_files=os.listdir(image_loc)
	image_files.sort()
	for file in image_files:
    		if not file.endswith('.jpg'):
        		image_files.remove(file)
	dx_img=1/fps
	time_img=np.array([i*dx_img for i in range(len(image_files))])
	image_index=np.array([i for i in range(len(image_files))])


	# load data
	temp_data=np.loadtxt(temp_loc, skiprows=23)
	dx_temp=temp_data[1,0]

	# convert temp to hf:
	time_temp=temp_data[:,0]
	avg_temps=np.average(temp_data[:,1:5], axis=0)
	temp_index=np.argsort(avg_temps)
	temp4=temp_data[:,temp_index[0]+1]
	temp3=temp_data[:,temp_index[1]+1]
	temp2=temp_data[:,temp_index[2]+1]
	temp1=temp_data[:,temp_index[3]+1]

	temp=np.transpose(np.array([temp1,temp2,temp3,temp4]))

	#Calculating heat flux
	tc_loc=np.array([0, 2.54, 5.08, 7.62])
	tc_loc=tc_loc*.001
	n=4
	k=392
	slope_d=n*np.sum(np.power(tc_loc,2))-np.sum(tc_loc)**2
	slope=(n*np.dot(temp,tc_loc)-np.sum(tc_loc)*np.sum(temp,axis=1))/slope_d
	hf=-k*slope/10000

    
	# interpolate data
	val=np.loadtxt(temp_loc,skiprows=10, max_rows=1,usecols=1, dtype=str)
	temp_start_min=float(str(val).split(':')[-2])
	temp_start_sec=float(str(val).split(':')[-1])
	print(temp_start_min,temp_start_sec)
	realtime_hf=time_temp+(60*temp_start_min)+temp_start_sec
	realtime_img=time_img+image_start_sec
	hf_match=np.interp(realtime_img,realtime_hf, hf)
	time=np.array([i*dx_img for i in range(len(image_files))])

	# Segment data into sequences 
	def split_data(signal,sl,st):
    		length=len(signal)
    		amt_samples=int((length-sl)/st)
    		data=np.empty((amt_samples,sl))
    		for i in range(amt_samples):
        		data[i]=signal[(i*st):(i*st)+sl]
        
    		return data


	def create_data(signal, target, time, sl, st):
    		seq=split_data(signal, sl, st )
    		y=split_data(target, sl, st)[:,-1]
    		t= split_data(time, sl, st)[:,-1]
    		return seq, y, t

	seq,y,t=create_data(image_index, hf_match, time, sl, st)

	
	print(seq.shape,y.shape,t.shape)
	chf=np.argwhere(y==np.max(y))
	cropval=int(chf[-1])
	intcropval=700
	seq=seq[intcropval:cropval]
	y=y[intcropval:cropval]
	t=t[intcropval:cropval]
	image_files[intcropval:cropval]
	#plt.plot(y)
	#plt.show()
	return seq,y,t,image_files

# Make a list of different file locations (image file location, image start time, temperature file location)

train_sets=[['/mnt/share/zdrive/Christy/Boiling-Data/87/Boiling-87/',32*60+52.526,'/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Sound/Temperature-87.lvm'],
['/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Images/Boiling-89/',29*60+22.0450,'/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Sound/Temperature-89.lvm'],
['/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Images/Boiling-90/',37*60+23.847735,'/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Sound/Temperature-90.lvm'],
	['/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Images/Boiling-91/',56*60+21.330747,'/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Sound/Temperature-91.lvm'],
	['/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Images/Boiling-92/',53*60+59.378147,'/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Sound/Temperature-92.lvm'],
	['/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Images/Boiling-94/',52*60+19.349909,'/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Sound/Temperature-94.lvm'],
	['/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Images/Boiling-96-2/',1*60+27.309121,'/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Sound/Temperature-96-2.lvm'],
	['/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Images/Boiling-96-3/',14*60+57.096798,'/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Sound/Temperature-96-3.lvm'],
	['/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Images/Boiling-96-4/',29*60+56.98783,'/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Sound/Temperature-96-4.lvm']]
	
val_set=[['/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Images/Boiling-96-1/',49*60+32.661077,'/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Sound/Temperature-96-1.lvm']]

test_set=[['/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Images/Boiling-93/',11*60+47.829513,'/mnt/share/zdrive/Christy/Acoustic-Data/Transient/Sound/Temperature-93.lvm']]


seqlen=1
stride=1
fps=150

#---------Train-Set------------
#seq=np.empty((0,seqlen))
image_paths=np.empty((0,seqlen),dtype=object)
y=np.empty((0))
t=np.empty((0))
save_folder='./'
for index in range(len(train_sets)):
	seq,y1,t1,image_files=data_gen(train_sets[index][0],train_sets[index][1],train_sets[index][2],fps,seqlen,stride)
	image_path=np.empty(seq.shape,dtype=object)
	for i in range(len(seq)):
		for j in range(len(seq[0])):
			image_path[i][j]=train_sets[index][0]+image_files[int(seq[i][j])]
	image_paths=np.append(image_paths, image_path, axis=0)
	y=np.append(y, y1, axis=0)
	t=np.append(t,t1, axis=0)
	

df=pd.DataFrame(y,columns=['Heat Flux'])
df['time']=t
for i in range((image_paths.shape[1])):
    df[f'image-{i}']=image_paths[:,i]

file_name=f'Train_full_set.csv'
df.to_csv(save_folder+file_name)

# ------------Val-Set----------------
image_paths=np.empty((0,seqlen),dtype=object)
y=np.empty((0))
t=np.empty((0))

for index in range(len(val_set)):
	seq,y1,t1,image_files=data_gen(val_set[index][0],val_set[index][1],val_set[index][2],fps,seqlen,stride)
	image_path=np.empty(seq.shape,dtype=object)
	for i in range(len(seq)):
		for j in range(len(seq[0])):
			image_path[i][j]=val_set[index][0]+image_files[int(seq[i][j])]
	image_paths=np.append(image_paths, image_path, axis=0)
	y=np.append(y, y1, axis=0)
	t=np.append(t,t1, axis=0)
	

df=pd.DataFrame(y,columns=['Heat Flux'])
df['time']=t
for i in range((image_paths.shape[1])):
    df[f'image-{i}']=image_paths[:,i]

file_name=f'Val_full_set.csv'
df.to_csv(save_folder+file_name)

# ------------Test-Set-------------------
image_paths=np.empty((0,seqlen),dtype=object)
y=np.empty((0))
t=np.empty((0))

for index in range(len(test_set)):
	seq,y1,t1,image_files=data_gen(test_set[index][0],test_set[index][1],test_set[index][2],fps,seqlen,stride)
	image_path=np.empty(seq.shape,dtype=object)
	for i in range(len(seq)):
		for j in range(len(seq[0])):
			image_path[i][j]=test_set[index][0]+image_files[int(seq[i][j])]
	image_paths=np.append(image_paths, image_path, axis=0)
	y=np.append(y, y1, axis=0)
	t=np.append(t,t1, axis=0)
	

df=pd.DataFrame(y,columns=['Heat Flux'])
df['time']=t
for i in range((image_paths.shape[1])):
    df[f'image-{i}']=image_paths[:,i]

file_name=f'Test_full_set.csv'
df.to_csv(save_folder+file_name)
