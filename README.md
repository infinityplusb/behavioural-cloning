# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


[//]: # (Image References)

[distribution]: ./initial_distribution.png "Distribution"
[images]: ./image_captures.png "Images"
[Nvidia]: ./9-layer-ConvNet-model.png "NVidia"

Overview
---
This project was to train a car to drive around the track after using a Convolution Neural Network to learn the correct steering patterns.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

The project includes, amongst other files:
* [model.ipynb](./model.ipynb) (notebook used to create and train the model)
* [drive.py](./drive.py) (unaltered script to drive the car - feel free to modify this file)
* [model.h5](./model.h5) (a trained Keras model)
* this report writeup file (it's markdown!!)
* [run1.mp4](./run1.mp4) (a video recording of your vehicle driving autonomously around the track for at least one full lap)

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The sample data should be available online, or you can generate your own data using the simulator.

To train the model, I used a "default" AWS Sagemaker notebook. The notebook was running on a p3 instance (with GPU). Create a notebook using `amazonei-tensorflow-p3` conda environment. You will need to install Keras. I found this easier to do via the terminal, than through the gui provided.     
So go, to terminal, and change into the proper conda environment.    
```
source activate amzonei-tensorflow-p36
conda install keras
```

### Drive notebook
#### Data Prep
The first 2 steps import the necessary libraries for processing.
The 3rd step, loads the files from disk into memory. It starts by loading the log file, a list of the turning and acceleration at each step of the way, and creates an array of the lines of this file.    
This particular step, in this instance, tries to average out the steering angles by using a moving average. This may be useful if you are using images that you train on a keyboard, as the steering angles aren't as smooth as if we a mouse or gamepad.    
As it happens, that step was unnecessary here, but is left in for future development.

One issue with using the moving averages was that the extreme steering angles are smoothed away, and so the car found it difficult to steer around the sharpest corner.
One way I would look to remedy that, would be to bring in the smoothed and unsmoothed steering, and bias the training towards using more of the unsmoothed images.

The unsmoothed steering angles are heavily biased around 0, i.e. driving straight.
![distribution][distribution]


The `for loop` in step 5 iterates through the steering files and reads in the related images. It adds in a correction (+/- 0.2) to the images if the file is reading in either the left or right camera.    
This data is appended to the final output set. The data is flipped and the steering angle reversed. This data is also appended onto the output set.   
No further alterations were done. Previously some experimentation with colour and brightness was trialled, but this was surprisingly less successful than just using the raw images.

The output images are fairly standard with no discerning changes to what's viewed via the simulator.

![images][images]

The model itself is the [NVidia model](https://arxiv.org/abs/1604.07316).
"The network consists of 9 layers, including an normalisation layer, 5 convolution layers and 3 fully connected layers."

![Nvidia][NVidia]

Using a larger number of layers, or using a 512 fully connnected layer, resulted in Out-Of-Memory errors while training.
Removing one of the fully connected layers meant the model didn't successfully steer around corners.

The Dropouts were used to reduce overfitting and overall network size.
An Adam Optimiser was chosen to enable a dynamic learning rate. The data was split using a random validation split of 0.05, as the problem was geared more to a "classification" type problem. With this in mind, the result of the training loss, and not just the validation loss, was considered.

### The Output
The model was able to successfully negotiate the track for at least 2 iterations, as per the output [video](run1.mp4). The first lap, the car ran onto the side of the track on the sharp right turn, however this was successfully cornered on the subsequent lap without touching the verge.

### TODO
Add in more images to train the model
Add in the speed component to enable the car to adjust speeds more dynamically, especially regarding straights and sharp corners.
Try the second track, and generate data for the second track. This has a larger variety of turns and twists and would be a greater challenge.
