from get_data import get_images, get_mapping
import sys
from keras.models import load_model
import matplotlib.pyplot as plt

try:
	predictions = []
	img_dir =sys.argv[1]
	img =get_images(path = img_dir)
	model = load_model('Data/Model/Model_save.h5')
	for pred in img:
		prediction = model.predict(pred).argmax(axis = 1)[0]
		prediciton = get_mapping(prediction)
    	predictions.append(prediction)
    print(f"The Captcha Text is {''.join(predictions)}")
except:
	print("Sorry!! No File provided for prediction.")
