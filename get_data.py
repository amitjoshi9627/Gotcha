import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import keras

def get_mapping(val = -1):
    if val == -1:
        return
    mapping = {'A': 8, 'B': 9, 'C': 10, 'D': 11, 'E': 12, 'F': 13, 'G': 14,
           'H': 15, 'J': 16, 'K': 17, 'L': 18, 'M': 19, 'N': 20, 'P': 21, 
           'Q': 22, 'R': 23, 'S': 24, 'T': 25, 'U': 26,'V': 27, 'W': 28,
           'X': 29, 'Y': 30, 'Z': 31, '2':0,'3': 1, '4': 2,'5': 3, '6':4,
           '7': 5,'8': 6,
           '9': 7}

    prediction = 8
    for key,value in mapping.items():
        if value == val:
            prediction = key
            break
    return prediction
def get_dataset():
    base_filename = 'extracted_letter_images'
    y = np.array([])
    len_letter = []
    images = []
    for folder in os.listdir(base_filename):
        Subdir = os.path.join(base_filename,folder)
        file_names = glob.glob( os.path.join( Subdir, '*' ))
        n_files = len(file_names)
        len_letter.append(n_files)
        label = mapping[folder]
        labels = np.full(n_files,label)
        
        if y.shape == (0,):
            y = labels
        else:
            y = np.append(y,labels)
        
        for name in file_names:
            image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
            images.append(image)

    """#### Resizing images"""

    img_width, img_height = 20, 20

    resized_images = [cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC) for img in images]

    X_train,X_test,y_train,y_test = train_test_split(np.array(resized_images),y, train_size=0.8, test_size=0.2)
    X_train = X_train.reshape(len(X_train),20,20,1).astype('float32')
    X_test = X_test.reshape(len(X_test),20,20,1).astype('float32')
    y_train = keras.utils.to_categorical(y_train, 32)
    y_test = keras.utils.to_categorical(y_test, 32)

    return X_train,X_test,y_train,y_test
    

def get_images(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img)
    image = cv2.copyMakeBorder(img,8,8,8,8,cv2.BORDER_REPLICATE)

    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if imutils.is_cv2() else contours[1]
    letter_image_regions = []

    for contour in contours:

    (x,y,w,h) = cv2.boundingRect(contour)
    if h == 0:
        print(x,y,w,h)
    if w/h > 1.25:

        half_width = int(w/2)
        letter_image_regions.append((x, y, half_width, h))
        letter_image_regions.append((x + half_width, y, half_width, h))

    else:
        letter_image_regions.append((x,y,w,h))

    if len(letter_image_regions)>4:
    print('Sorry! but the captcha is more than 4 letters!!')

    else:
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    llist = []
    for box,text in zip(letter_image_regions, correct_text):

        (x,y,w,h) = box

        letter_image = image[y-2: y+h+2,x-2:x+w+2]

        X = cv2.resize(letter_image, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
        
        X = X.reshape(1,20,20,1).astype('float32')
        llist.append(X)
    return llist