from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split



# LOADING DATA

TRAIN_PATH = 'chest_xray/test'
TEST_PATH = 'chest_xray/train'

normal_files = glob(TRAIN_PATH + "/NORMAL/*.jpeg")
normal_files.extend(glob(TRAIN_PATH + "/NORMAL/*.jpg"))
normal_files.extend(glob(TRAIN_PATH + "/NORMAL/*.png"))
sick_files = glob(TRAIN_PATH + "/PNEUMONIA/*.jpeg")
sick_files.extend(glob(TRAIN_PATH + "/NORMAL/*.jpg"))
sick_files.extend(glob(TRAIN_PATH + "/NORMAL/*.png"))

normal_data = []
sick_data = []

for file in normal_files:
    image = cv2.imread (file)
    normal_data.append (image)

for file in sick_files:
    image = cv2.imread (file)
    sick_data.append (image)


# SHOW IMAGES EXAMPLE AND INITIAL DATA

print("Number of healthy patients images: " + str(len(normal_data)))
print("Number of sick patients images: " + str(len(sick_data)))

cv2.imshow("Healthy 1", normal_data[0]) 
cv2.imshow("Healthy 2", normal_data[1]) 
cv2.imshow("Healthy 3", normal_data[2]) 

print("Healthy image 1 shape: " + str(normal_data[0].shape))
print("Healthy image 2 shape: " + str(normal_data[1].shape))
print("Healthy image 3 shape: " + str(normal_data[2].shape))

cv2.imshow("Sick 1", sick_data[0]) 
cv2.imshow("Sick 2", sick_data[1]) 
cv2.imshow("Sick 3", sick_data[2]) 

print("Sick image 1 shape: " + str(sick_data[0].shape))
print("Sick image 2 shape: " + str(sick_data[1].shape))
print("Sick image 3 shape: " + str(sick_data[2].shape))


# EXPLORATION (colors, what else?)

# CONVERT TO GREYSCALE

normal_image = normal_data[0]
gray_image = cv2.cvtColor(normal_image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Image before conversion", normal_image) 
cv2.imshow("Image after conversion", gray_image) 

print("Before conversion shape: " + str(normal_data[0].shape))
print("After conversion shape: " + str(gray_image.shape))
# TODO: Hardly affects size?

for index, image in enumerate(normal_data):
    normal_data[index] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
for index, image in enumerate(sick_data):
    sick_data[index] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# COMPRESS
for index, image in enumerate(normal_data):
    normal_data[index] = cv2.resize(image, (170,120))
    
for index, image in enumerate(sick_data):
    sick_data[index] = cv2.resize(image, (170,120))
    

# NORMALIZE VALUES
for index, image in enumerate(normal_data):
    normal_data[index] = image / 255
    
for index, image in enumerate(sick_data):
    sick_data[index] = image / 255


# CREATE Y [0,1]
    
normal_Y = [0]*len(normal_data)
sick_Y = [1]*len(sick_data)

# [sick, sick, sick, normal]
# [1, 1, 1, 0]


for index, image in enumerate(normal_data):
    normal_data[index] = image.flatten()
    
for index, image in enumerate(sick_data):
    sick_data[index] = image. flatten()

train_images = np.array(normal_data + sick_data)
images_labels = np.array(to_categorical(normal_Y + sick_Y))
X_train, X_test, y_train, y_test = train_test_split(train_images, images_labels, test_size=0.2)



model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(20400,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=128,
                    epochs=30,
                    verbose=1,
                    shuffle=True,
                    validation_data=(X_test, y_test))


score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



#LR_reduce=keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
#                            factor=.5,
#                            patience=10,
#                            min_lr=.000001,
#                            verbose=0)
#
#ES_monitor=keras.callbacks.EarlyStopping(monitor='val_loss',
#                          patience=20)

# FIT INTO MODEL
#
#
#model = Sequential()
#model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))