# Code For Training curves plots
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

def train_plot(file_path):
	data=np.load(file_path)
	plt.plot(data[0], label='Loss')
	plt.plot(data[1], label='Val Loss')
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()
	

arial_font_path='/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
arial=font_manager.FontProperties(fname=arial_font_path)

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Plot the figures on the grid
data=np.load('./Shuffle-1/loss_cnn.npy')
axs[0, 0].plot(data[0] , label='Loss')
axs[0,0].plot(data[1], label='Val Loss')
axs[0,0].legend()
axs[0, 0].set_title('CNN',fontproperties=arial, fontsize=12)

data=np.load('./Shuffle-1/pca900-mlp-static-loss.npy')
axs[0, 1].plot(data[0] , label='Loss')
axs[0,1].plot(data[1], label='Val Loss')
axs[0,1].set_title('PCA',fontproperties=arial, fontsize=12)

data=np.load('./loss_pc40_sl200_lstm.npy')
axs[1, 0].plot(data[0] , label='Loss')
axs[1,0].plot(data[1], label='Val Loss')
axs[1, 0].set_title('LSTM',fontproperties=arial, fontsize=12)

data=np.load('./Shuffle-1/loss_fft-pc40_sl200.npy')
axs[1, 1].plot(data[0] , label='Loss')
axs[1,1].plot(data[1], label='Val Loss')
axs[1, 1].set_title('PCA-FFT-CNN',fontproperties=arial, fontsize=12)

data=np.load('./Shuffle-1/loss_pc40_sl200.npy')
axs[1, 2].plot(data[0] , label='Loss')
axs[1,2].plot(data[1], label='Val Loss')
axs[1, 2].set_title('PCA-CNN',fontproperties=arial, fontsize=12)

# Hide the remaining empty subplots
axs[0, 2].axis('off')

# Adjust the spacing between the subplots
plt.tight_layout()

# Show the plots
plt.show()
