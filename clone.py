import csv
import cv2
import os
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf

#import keras as K
#gpus = K.get_session().list_devices() 
#run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

input_path = '/home/brian/Documents/Training/Udacity/Self_Driving_Car/beta_simulator_linux/beta_simulator_Data/'
image_path = input_path + 'IMG/'

lines = []
with open(os.path.join(input_path, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.utils import shuffle

import sklearn
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.1)
train_samples = train_samples[1:] 

#### Split out the data into seperate train and valid directories
import shutil
def move_files_to_folder(my_images, image_type) :
    location_to_make = './' + image_type + 'ing_images'
    if not os.path.exists(location_to_make):
        os.mkdir(location_to_make)
        
    if location_to_make.count('./') > 1:
        shutil.rmtree(location_to_make, ignore_errors=False)
        os.makedirs(location_to_make)
        print("Successfully cleaned directory " + location_to_make)
     
    for images in my_images :
        for i in range(3) :
#            print(i)
#            print(images[i])
#            print(location_to_make + '/')
            shutil.copy(images[i], location_to_make + '/')

#move_files_to_folder(train_samples, 'train')
#move_files_to_folder(validation_samples, 'valid')

from keras.preprocessing.image import ImageDataGenerator
        
datagen = ImageDataGenerator(
        validation_split=0.2)

#        rotation_range=40,
#        width_shift_range=0.2,
#        height_shift_range=0.2,
#        rescale=1./255,
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True,
#        fill_mode='nearest')

train_generator = datagen.flow_from_directory(
        image_path , subset='training')

val_generator = datagen.flow_from_directory(
        image_path , subset='validation')


def generator(samples, type1, batch_size = 32):
    num_samples = len(samples)
    if type1 == 'valid' :
        indices = np.random.randint(0, batch_size, num_samples)
        samples = np.array(samples)[indices]
    correction = 0.2
    
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):

            batch_images = samples[offset:offset+batch_size]

            images = []
            measurements = []

            for line in batch_images:
                # do the centre image
                image = cv2.imread(line[0])
                measurement = np.round(np.float16(line[3]),2) + .01
                images.append(image)
                measurements.append(measurement)
                # flip the centre image
                images.append(cv2.flip(image, 1))
                measurements.append(measurement*-1.0)

                # do the left image
                image = cv2.imread(line[1])
                measurement = np.float16(line[3]) + correction
                images.append(image)
                measurements.append(measurement)
                # flip the left image
                images.append(cv2.flip(image, 1))
                measurements.append(measurement*-1.0)

                # do the right image
                image = cv2.imread(line[2])
                measurement = np.float16(line[3]) - correction
                images.append(image)
                measurements.append(measurement)
                # flip the right image
                images.append(cv2.flip(image, 1))
                measurements.append(measurement*-1.0)
                
                X_train = np.array(images)
                y_train = np.array(measurements)
                yield shuffle(X_train, y_train)


batch_size = 32

train_generator = generator(train_samples, type1='train', batch_size=batch_size)
validation_generator = generator(validation_samples, type1='valid', batch_size=batch_size)

from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import BatchNormalization
from keras.layers import ELU
from keras.layers import Dropout

with tf.device("/cpu:0"):
    model = Sequential()

model = Sequential()
model = multi_gpu_model(model, gpus=2, cpu_relocation=True)
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3))) #, output_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Conv2D(24,(5,5), activation='relu'))
model.add(ELU())
model.add(BatchNormalization())
model.add(Conv2D(36,(5,5), activation='relu'))
model.add(ELU())
model.add(BatchNormalization())
model.add(Conv2D(48,(5,5), activation='relu'))
model.add(ELU())
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(ELU())
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(ELU())

#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(ELU())

model.add(Dense(512))
model.add(Dropout(0.6))
model.add(ELU())

model.add(Dense(100))
model.add(Dropout(0.6))
model.add(ELU())

model.add(Dense(10))
model.add(Dropout(0.6))
model.add(ELU())


# Batch Normalisation
#model.add(BatchNormalization())

model.add(Dense(1))

#with tf.device("/gpu:1"):
model.compile(loss='mse', optimizer='adam')#, options = run_opts)

model.fit_generator(train_generator
                    #, steps_per_epoch=20 
                    , np.ceil(len(train_samples)/batch_size)
                    , validation_data=validation_generator
                    , validation_steps=np.ceil(len(validation_samples)/batch_size)
                    #, use_multiprocessing=True
                    #, workers = 4
                    , epochs=5
                    , verbose=1)

print(np.ceil(len(train_samples)/batch_size))
print(np.ceil(len(validation_samples)/batch_size))
#model.fit(X_train, y_train, validation_split=0.4, shuffle=True, epochs=10)

model.save('model_09.h5')

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