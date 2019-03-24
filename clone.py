import csv
import cv2
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf

input_path = '/home/brian/Documents/Training/Udacity/Self_Driving_Car/beta_simulator_linux/beta_simulator_Data/'
image_path = input_path + 'IMG/'

lines = []
with open(os.path.join(input_path, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    correction = 0.4
    filename = source_path.split('/')[-1]
    current_path = os.path.join(os.path.join(input_path, 'IMG/', filename))
    image = cv2.imread(current_path)
    images.append(image)
    if 'left' in current_path:
        measurement = np.round(float(line[3]), decimals=2) + correction
    elif 'right' in current_path:
        measurement = np.round(float(line[3]), decimals=2) - correction
    else :
        measurement = np.round(float(line[3]), decimals=2)
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    random_bright = .25+np.random.uniform() 

    augmented_images.append(image)
    augmented_measurements.append(measurement)

    # add brightness
    bright_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    bright_image[:,:,2] = bright_image[:,:,2]*random_bright       
    augmented_images.append(bright_image)
    augmented_measurements.append(measurement)
 
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
    augmented_images.append(cv2.flip(bright_image,1))
    augmented_measurements.append(measurement*-1.0)
     



X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import BatchNormalization
from keras.layers import ELU
from keras.layers import Dropout

with tf.device("/cpu:0"):
    model = Sequential()

model = Sequential()
#model = multi_gpu_model(model, gpus=2, cpu_relocation=True)
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3))) #, output_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Conv2D(24,(5,5), activation='relu'))
#model.add(ELU())
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(36,(5,5), activation='relu'))
#model.add(ELU())
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(48,(5,5), activation='relu'))
#model.add(ELU())
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3), activation='relu'))
#model.add(ELU())
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3), activation='relu'))
#model.add(ELU())
model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
#model.add(ELU())

model.add(Dense(100))
model.add(Dropout(0.4))
model.add(ELU())

#model.add(Dense(50))
#model.add(Dropout(0.4))
#model.add(ELU())

model.add(Dense(10))
model.add(Dropout(0.4))
model.add(ELU())

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')#, options = run_opts)

history_object = model.fit(x=X_train, y=y_train
                           , batch_size = 8
                           , verbose=1
                           , validation_split=0.01
                           , epochs=10)

#model.fit(X_train, y_train, validation_split=0.4, shuffle=True, epochs=10)

model.save('model_15.h5')

import matplotlib.pyplot as plt

#history_object = model.fit_generator(train_generator, samples_per_epoch =
#    len(train_samples), validation_data = 
#    validation_generator,
#    nb_val_samples = len(validation_samples), 
#    nb_epoch=5, verbose=1)

## print the keys contained in the history object
print(history_object.history.keys())

## plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()