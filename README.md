Wheat-quality-detector-2
========================
Inspired by [link](https://github.com/dhishku/Machine-Learning-for-Grain-Assaying)

Description:
-----------

This wheat quality detection test we are using to identify the quality of given wheat grains image. The dataset we used is provided by [link](https://github.com/dhishku/Machine-Learning-for-Grain-Assaying). 

The wheat quality detection problem is divided into two sub problems given as following:
1. Two classs classification i.e. a healthy grain or other.
2. Five class slassification i.e. a healthy grain, damaged grain, foreign partical, broken grain and grain cover.

The dataset we used for training (is the single grain or other images) extracted for above mentioned dataset.

Requirement:
-----------
- opencv-python
- keras
- tensorflow
- matplotlib

Tested with **python3.5**

A Glance
--------
For two class classification:

     $ python classifier_2_v2.py
     68/68 [==============================] - 0s 499us/step - accuracy: 0.9020 - loss: 0.2475
     MLP Test loss: 0.247524231672287
     MLP Test accuracy: 0.9019879698753357

For five class classification:

     $ python classifier_5_v2.py
     65/65 [==============================] - 0s 532us/step - loss: 0.4837 - accuracy: 0.8254
     MLP Test loss: 0.483661413192749
     MLP Test accuracy: 0.8253890872001648

For performing a saimple test:

     $ python cmd_wheat_quality_detector_v2.py
     Enter the file(wheat image) location to dectect : test_2.jpg
     Segmentation in process...
     Level 1 segmentation Finished:
     Rejected segment: 1
     Level 2 segmentation Finished:
     Rejected segment: 21

     Total number of segments 124
     Number of rejected segments 22

     Segmentation in Complete.

     Feature extraction in process...
     Feature extraction in complete.

     Number of good grain : 84
     Number Not good grain or imputity: 18

About:
----
Please feel free to [email & contact me](mailto:deepaksinghrana049@gmail.com) if you run into issues or just would like to talk about the future usage.

