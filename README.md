# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.py from imgs to make a mp4 video
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 133-151) 

The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer (code line 125). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 29). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 176).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road .

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use NVIDIA's model.

#### 2. Final Model Architecture

The final model architecture (model.py lines 133-151) consisted of a convolution neural network with the following layers:

|Layer         |          Output Shape       |       Param   |
|:---------------------:|:---------------------------------------------:|:---------------------:|
|lambda_1 (Lambda)          |  (None, 160, 320, 3)   |    0          |
| cropping2d_1 (Cropping2D)  |  (None, 90, 320, 3)   |     0          |
| conv2d_1 (Conv2D)          |  (None, 86, 316, 24)  |     1824       |
| max_pooling2d_1 (MaxPooling2 | (None, 43, 158, 24) |      0          |
| activation_1 (Activation)  |  (None, 43, 158, 24)  |     0          |
| conv2d_2 (Conv2D)          |  (None, 39, 154, 36)  |     21636      |
| max_pooling2d_2 (MaxPooling2 | (None, 20, 77, 36)  |      0          |
| activation_2 (Activation)  |  (None, 20, 77, 36)   |     0          |
| conv2d_3 (Conv2D)          |  (None, 16, 73, 48)   |     43248      |
| max_pooling2d_3 (MaxPooling2 | (None, 8, 37, 48)   |      0          |
| activation_3 (Activation)  |  (None, 8, 37, 48)    |     0          |
| conv2d_4 (Conv2D)          |  (None, 6, 35, 64)    |     27712      |
| max_pooling2d_4 (MaxPooling2 | (None, 3, 18, 64)   |      0          |
| activation_4 (Activation)  |  (None, 3, 18, 64)    |     0          |
| conv2d_5 (Conv2D)          |  (None, 1, 16, 64)    |     36928      |
| max_pooling2d_5 (MaxPooling2 | (None, 1, 8, 64)    |      0          |
| activation_5 (Activation)  |  (None, 1, 8, 64)     |     0          |
| flatten_1 (Flatten)        |  (None, 512)          |     0          |
| dense_1 (Dense)            |  (None, 1164)         |     597132     |
| activation_6 (Activation)  |  (None, 1164)         |     0          |
| dense_2 (Dense)            |  (None, 100)          |     116500     |
| activation_7 (Activation)  |  (None, 100)          |     0          |
| dense_3 (Dense)            |  (None, 50)           |     5050       |
| activation_8 (Activation)  |  (None, 50)           |     0          |
| dense_4 (Dense)            |  (None, 10)           |     510        |
| activation_9 (Activation)  |  (None, 10)           |     0          |
| dense_5 (Dense)            |  (None, 1)            |     11         |
| 


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

<img src="examples/nvidiamodel.png" width="500" />

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

<img src="examples/center.jpg" width="500" />

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to keep itself in the center. 
These images show what a recovery looks like starting from right to center :

<div class="test">
<img src="examples/center_2018_05_19_10_44_10_311.jpg" width="200" />
<img src="examples/center_2018_05_19_10_44_10_411.jpg" width="200" />
<img src="examples/center_2018_05_19_10_44_10_516.jpg" width="200" />
</div>
 


To augment the data sat, I also run the vehicle in the reverse direction:

<div class="test">
<img src="examples/cropbefor.jpg" width="200" />
<img src="examples/cropafter.jpg" width="200" />

</div>

After the collection process, I had 55887 number of data points.  


I finally randomly shuffled the data set and put 80% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by loss and acc. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.
