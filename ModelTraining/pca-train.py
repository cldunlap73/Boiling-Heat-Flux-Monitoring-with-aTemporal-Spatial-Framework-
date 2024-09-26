'''
This Code Preforms PCA Transformation
Since the dataset is large different amounts of images will be used to gauge the smallest
subset of data that can be used in the transformation with similar results.
'''
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import random
import pickle

# load pandas dataframe from csv
csv_loc="./Train_full_set.csv"
save_loc="./pca.pkl"
df=pd.read_csv(csv_loc)
# Define random array
amount=int(len(df)*.7)
indexs=[i for i in range(amount)]
random.shuffle(indexs)
#np.save('./pca_random_shuffle.npy',np.array(indexs))
splits=[amount]
explained_variances=[]
for split in splits:
	# load images into memory
	image_array=np.empty((split, 200*200))
	for i,idx in enumerate(indexs[:split]):
		image_path=df.loc[idx,'image-0']
		image=Image.open(image_path)
		image=image.resize((200,200))
		image=np.reshape(np.array(image)/255., (200*200))
		image_array[i]=image
	pca=PCA(900)
	pca.fit(image_array)
	
	with open(save_loc, 'wb') as file:
		pickle.dump(pca,file)
			
		
	
