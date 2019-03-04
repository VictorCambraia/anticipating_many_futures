# Anticipating Many Futures: Online Human Motion Prediction and Generation for Human-Robot Interaction
Code for https://ieeexplore.ieee.org/abstract/document/8460651


This code contains code to track a human skeleton using a Kinect sensor and to predict / sample future trajectories using a conditional variational autoencoder. 

# Setup
The underlying tracking software is OpenNI and underlies the copyright of PrimeSense Ltd.    
To run this code, make sure to have installed all dependencies for OpenNI / PrimeSense and that everything is running with a Kinect V1 sensor.


# Use online
The main action happens in SkeletonTracker.cpp, which records and processes the pose data online and in Vaecoder.cpp, which is responsible for prediction. The integrator.cpp file makes sure that these two processes are synced. 

# Python training
The model training is done in python, using theano. The basic code can be found in the python_model_training folder.




