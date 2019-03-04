# Anticipating Many Futures: Online Human Motion Prediction and Generation for Human-Robot Interaction
https://ieeexplore.ieee.org/abstract/document/8460651


This code contains code to track a human skeleton using a Kinect sensor and to predict / sample future trajectories using a conditional variational autoencoder. 

The underlying tracking software is OpenNI and underlies the copyright of PrimeSense Ltd.    
To run this code, make sure to have installed all dependencies for OpenNI / PrimeSense and that everything is running with a Kinect V1 sensor.

The main action happens in SkeletonTracker.cpp, which records and processes the pose data online and in Vaecoder.cpp, which is responsible for prediction. The integrator.cpp file makes sure that these two processes are synced. 



