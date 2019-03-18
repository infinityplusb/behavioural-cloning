import csv
import cv2
import os
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#import tensorflow as tf

input_path = '/home/brian/Documents/Training/Udacity/Self_Driving_Car/beta_simulator_linux/beta_simulator_Data/'

lines = []
with open(os.path.join(input_path, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        

#for line in lines:
from sklearn.utils import shuffle

import sklearn
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print(len(train_samples))
print(len(validation_samples))

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    #print(num_samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_images = samples[offset:offset+batch_size]
            #print("number of images in this batch: ", len(batch_images))
            images = []
            measurements = []
            for line in batch_images:
                correction = 0.2
                # do the centre image
                image = cv2.imread(line[0])
                measurement = float(line[3])
                images.append(image)
                measurements.append(measurement)
                # flip the centre image
                images.append(cv2.flip(image, 1))
                measurements.append(measurement*-1.0)

                # do the left image
                image = cv2.imread(line[1])
                measurement = float(line[3]) + correction
                images.append(image)
                measurements.append(measurement)
                # flip the left image
                images.append(cv2.flip(image, 1))
                measurements.append(measurement*-1.0)

                # do the centre image
                image = cv2.imread(line[2])
                measurement = float(line[3]) - correction
                images.append(image)
                measurements.append(measurement)
                # do the centre image
                images.append(cv2.flip(image, 1))
                measurements.append(measurement*-1.0)
            
            #print(offset)
            #print("Length: " , len(measurements))
            X_train = np.array(images)
            y_train = np.array(measurements)
            #print(X_train.shape)
            #print(y_train.shape)
                #print("Taking a break at %d: " , offset)
            yield sklearn.utils.shuffle(X_train, y_train)

batch_size = 128

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)    

#augmented_images, augmented_measurements = [], []
#for image, measurement in zip(images, measurements):
#    augmented_images.append(image)
#    augmented_measurements.append(measurement)
#    augmented_images.append(cv2.flip(image,1))
#    augmented_measurements.append(measurement*-1.0)


#X_train = np.array(augmented_images)
#y_train = np.array(augmented_measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
#from keras.layers import BatchNormalization
from keras.layers import Dropout

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3))) #, output_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Conv2D(24,(5,5), activation='relu')) #, strides=(2,2)))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#model.add(BatchNormalization())

model.add(Conv2D(36,(5,5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#model.add(BatchNormalization())
#model.add(Conv2D(24,(5,5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#model.add(BatchNormalization())

#model.add(Conv2D(36,(5,5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#model.add(BatchNormalization())

model.add(Conv2D(48,(5,5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#model.add(BatchNormalization())

model.add(Conv2D(64,(3,3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(64,(3,3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#model.add(BatchNormalization())

model.add(Flatten())


model.add(Dense(1000))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
#model.add(BatchNormalization())


model.add(Dense(500))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
#model.add(BatchNormalization())

model.add(Dense(50))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
#model.add(BatchNormalization())


model.add(Dense(1))

#with tf.device("/gpu:1"):
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=np.ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)
#model.fit(X_train, y_train, validation_split=0.4, shuffle=True, epochs=10)

model.save('model_06.h5')

import matplotlib.pyplot as plt

#history_object = model.fit_generator(train_generator, samples_per_epoch =
#    len(train_samples), validation_data = 
#    validation_generator,
#    nb_val_samples = len(validation_samples), 
#    nb_epoch=5, verbose=1)

### print the keys contained in the history object
#print(history_object.history.keys())

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()