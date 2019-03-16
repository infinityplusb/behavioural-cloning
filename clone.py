import csv
import cv2
import os
import numpy as np

input_path = '/home/brian/Documents/Training/Udacity/Self_Driving_Car/beta_simulator_linux/beta_simulator_Data/'

lines = []
with open(os.path.join(input_path, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    source_path = line[0]
    correction = 0.2
    filename = source_path.split('/')[-1]
    current_path = os.path.join(os.path.join(input_path, 'IMG/', filename))
    image = cv2.imread(current_path)
    images.append(image)
    if 'left' in current_path:
        measurement = float(line[3]) + correction
    elif 'right' in current_path:
        measurement = float(line[3]) - correction
    else :
        measurement = float(line[3])
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D


model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(6,(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(6,(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.5, shuffle=True, epochs=1)

model.save('model_02.h5')