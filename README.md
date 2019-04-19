# Captcha_Solver
### By Amit Joshi

A Deep Learning model which tells you the letters of the captcha texts.
#### Example:
| <img src="captcha.jpg?raw=true" width="200">|

|Text: 8746|
### Descrition
The goal of the model is to extract all the letters of a captcha image.

### Dataset
The dataset contains about 10000 captcha images.

### Process
The following steps are done:
* Extracting all single letters from images and storing them. `python3 extract_single_letters.py`
* Making a neural network and training it on single letters.
* Doing the predictions.

### Model training
* Run `python3  train.py`
### Predictions
* Run `python3 predict.py <filename>`

### Notes
* Computing: Google Colab Tesla K80 GPU
* Python version: 3.6.6
* Using packages
  1. [`Keras`](https://www.tensorflow.org/guide/keras) (tensorflow.python.keras) for building models 
  2. [`OpenCV`](https://opencv.org/) (cv2) for processing images
  3. [`sikit-learn`](http://scikit-learn.org/stable/) (sklearn) for train_test_split 
  4. Install necessary modules with `sudo pip3 install -r requirements.txt` command.
