from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Dropout
import keras
import numpy as np
import matplotlib.pyplot as plt
import os

def save_model(model):
	if not os.path.exists('Data/'):
		os.makedirs('Data/')
	saving_path = os.path.join('Data/','Model_save.h5')
	model.save(saving_path)
	return

def get_model(num_classes = 32):
	model = Sequential()

	model.add(Conv2D(32,(3,3),input_shape = (20,20,1),activation = 'relu', name = 'Conv2D_1'))
	model.add(Conv2D(32,(3,3),activation = 'relu', name = 'Conv2D_2'))
	model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool2D_1'))
	model.add(Dropout(0.25, name = 'Dropout_1'))

	model.add(Conv2D(64,(3,3),activation = 'relu', name = 'Conv2d_3'))
	model.add(Conv2D(64,(3,3),activation = 'relu', name = 'Conv2d_4'))
	model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool2D_2'))
	model.add(Flatten(name = 'Flatten'))

	model.add(Dense(256,activation='relu', name = 'Dense_1'))
	model.add(Dropout(0.5, name = 'Dropout_2'))
	model.add(Dense(64,activation='relu',name = 'Dense_2'))
	model.add(Dense(num_classes,activation='softmax', name = 'Dense_3'))

	model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])

	print(model.summary())

	return model

if __name__ == '__main__':
	save_model(get_model())