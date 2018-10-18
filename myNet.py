from utils.mnist_reader import load_mnist
#import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import os
import numpy as np

DATADIR			= 'D:\\general\\techies\\computerVision\\vision projects\\fashion_MNIST\\fashion-mnist-master\\data\\fashion'
IPIMG_H			= 28
IPIMG_W			= 28
LBLCNT			= 10


def main():
	#Load data from dataset
	X_train, y_train	= load_mnist(DATADIR,'train') #X_train=60000 images, each 28x28; y_train = 60000 labels
	X_test, y_test		= load_mnist(DATADIR,'t10k')	#X_train=10000 images, each 28x28; y_train = 10000 labels
	X_train						= X_train.reshape(X_train.shape[0],IPIMG_H,IPIMG_W,1)
	#y_train						= y_train.reshape(y_train.shape[0],IPIMG_H,IPIMG_W)
	X_test						= X_test.reshape(X_test.shape[0],IPIMG_H,IPIMG_W,1)
	#y_test						= y_test.reshape(y_test.shape[0],IPIMG_H,IPIMG_W)
	
	
	#Construct a model
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), data_format='channels_last',input_shape=(IPIMG_H,IPIMG_W,1),activation='relu'))
	model.add(Conv2D(32,(3,3),activation='relu'))
	model.add(Conv2D(32,(3,3),activation='relu'))
	model.add(Conv2D(32,(3,3),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(LBLCNT, activation='softmax'))

	model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
	model.fit(
        X_train, y_train,
        epochs=5,
        validation_data=(X_test, y_test))
        #steps_per_epoch=10,
        #validation_steps=100)
        
	score = model.evaluate(X_test, y_test, steps=50)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
if __name__ == '__main__':
	main()
