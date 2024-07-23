import cv2
import os
from PIL import Image
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical


'''
STEP 1: CREATING THE DATASET: 
'''

dataset = [] #intialize dataset
label = [] #intialize label to keep track of which images are yes and no

#just for clarity
INPUT_SIZE = 64 
image_directory = 'datasets/' 

#access the images by entering the directory
no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')

# add all no images into the dataset
for i, image_name in enumerate(no_tumor_images): #iterate through all the no images
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'no/' + image_name) #loads image from a specified file
        image = Image.fromarray(image, "RGB") #creates an rgb image using an array of pixels
        image = image.resize((INPUT_SIZE, INPUT_SIZE)) # sets the image 64 by 64 pixels
        dataset.append(np.array(image)) #add the nparray to the dataset
        label.append(0) #0 stands for no
   
# add all yes images into the dataset    
for i, image_name in enumerate(yes_tumor_images): #iterate through all the yes images
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'yes/' + image_name) #loads image from a specified file
        image = Image.fromarray(image, 'RGB') #creates an rgb image using an array of pixels
        image = image.resize((INPUT_SIZE, INPUT_SIZE)) # sets the image 64 by 64 pixels
        dataset.append(np.array(image)) #add the nparray to the dataset
        label.append(1) #1 stands for no

dataset = np.array(dataset)
label = np.array(label)



'''
STEP 2: ESTABLISH THE TRAINING AND TEST SPLITS
'''

#create the split: train is 80% and test is 20%
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

#normalize the data
x_train = normalize(x_train, axis = 1)
x_test = normalize(x_test, axis = 1)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)



'''
STEP 3: BUILDING THE MODEL
'''

model = Sequential() 

model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3))) #extract features on a 3 RBG 64 x 64 images
model.add(Activation('relu')) #learn complex patterns?
model.add(MaxPooling2D(pool_size=(2,2))) #gathers only the important pixels for indentification

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #reshapes data into 1-D Array
model.add(Dense(64))  #
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size = 16, verbose = 1, epochs = 10, validation_data=(x_test, y_test), shuffle = False)

model.save = ('BrainTumor10EpochsCategorical.h5')