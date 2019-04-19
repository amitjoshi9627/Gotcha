from get_data import get_dataset
from get_model import get_model, save_model
import os
import numpy as np
import matplotlib.pyplot as plt

def train_model(model,X_train,X_test,y_train,y_test,batch_size = 32,num_epochs = 15):

	model.fit(X_train,y_train,batch_size=batch_size,validation_data=(X_test, y_test), epochs=num_epochs, verbose=1)

	return model

X_train,X_test,y_train,y_test = get_dataset()

model = get_model(num_classes = 32)

model = train_model(model, X_train, X_test, y_train, y_test,batch_size = 32,num_epochs = 15)

save_model(model)
