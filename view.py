import matplotlib.pyplot as plt
import numpy as np


data = np.load('training_data20000.npy')
seq = data[999]
fig = plt.figure(figsize=(9,9))

start_step = 0    #Change to start position if viewing small portion of data 
steps = len(seq) #Change to smaller constant to limit data view
rows = len(seq[start_step])*2 - 1


for i in range(steps):
	for j in range(rows):
		if j == 0:
			img = seq[start_step+i][0]
		else:
			img = seq[start_step+i][(j+1)/2][np.abs(j%2-1)]
		ax = fig.add_subplot(rows, steps, j*steps+(i+1)) 
		ax.imshow(img, cmap='gray')
		ax.axis('off')
plt.show()
