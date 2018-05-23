import csv

import cv2
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import *
from keras.initializers import *

from keras import backend as K
K.clear_session()


samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        #if steering value large than 0.6 , do not use to train set 
        if abs(float(line[3])) <= 0.98:
            line[0] = line[0].replace('\\','/')
            line[1] = line[1].replace('\\','/')
            line[2] = line[2].replace('\\','/')
            samples.append(line)
       

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# Data generator 
# Because the generator is slower than load all data in the momery, so not to use here
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)


# Getdata by driving_log.csv
def getdata(samples):
    sampleslen = len(samples) * 3
    correction = 0.2
    images = []
    angles = []
    num = 0
    for batch_sample in samples:
        #center
        name = './IMG/'+batch_sample[0].split('/')[-1]
        center_image = cv2.imread(name)
        center_angle = float(batch_sample[3])
        images.append(center_image)
        angles.append(center_angle)
        num += 1

        if (num % 1000 == 0):
            print('Loaded ', num , '	', num * 1.0 / sampleslen)


    for batch_sample in samples:
        #left
        name = './IMG/'+batch_sample[1].split('/')[-1]
        left_image = cv2.imread(name)
        left_angle = float(batch_sample[3]) + correction
        images.append(left_image)
        angles.append(left_angle)
        num += 1

        if (num % 1000 == 0):
            print('Loaded ', num , '	', num * 1.0 / sampleslen)


    for batch_sample in samples:
        #right
        name = './IMG/'+batch_sample[2].split('/')[-1]
        right_image = cv2.imread(name)
        right_angle = float(batch_sample[3]) - correction
        images.append(right_image)
        angles.append(right_angle)
        num += 1

        if (num % 1000 == 0):
            print('Loaded ', num , '	', num * 1.0 / sampleslen)

    # trim image to only see section with road
    X_train = np.array(images)
    y_train = np.array(angles)

    return X_train, y_train



#Get train and vlidation set
print('Loading train data , total ', len(train_samples) * 3)
X_train, y_train = getdata(train_samples)
print()
print('Loading train data , total ', len(validation_samples) * 3)
X_valid, y_valid = getdata(validation_samples)


#Trimmed image format
ch, row, col = 3, 160, 320 


#Use NVIDIA's model
model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))

#Crop image
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(row, col, ch)))

# Add Convolution layers
model.add(Conv2D(24, (5, 5)))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(Activation('relu'))

model.add(Conv2D(36, (5, 5)))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5)))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(Activation('relu'))

# Add flatten layer 
model.add(Flatten())

# Add fully connection layer
model.add(Dense(1164, kernel_initializer = TruncatedNormal(mean=0.0, stddev=0.1, seed=None), bias_initializer='zeros'))
model.add(Activation('relu'))

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

# Output layer
model.add(Dense(1))

# Print model summary
model.summary()

# Compile model
model.compile(loss='mse', optimizer='adam')

# Fit model
model.fit(X_train, y_train, batch_size=64, epochs=2, validation_data=(X_valid, y_valid), shuffle=True) 

# Save model
model.save('model.h5')





