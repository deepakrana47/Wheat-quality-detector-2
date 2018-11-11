# Wheat-quality-detector-2

In the wheat quality detection test we already segmented the wheat grains out of dataset provided in form of images containing multiple wheat grain in one picture by(https://github.com/dhishku/Machine-Learning-for-Grain-Assaying). The extracted wheat grains and other particals are than divided in five classes as follows:

>> Grain :- Contains the healthy wheat grains.

>> Damaged_grain :- Contain non healthy or deformed wheat grains. 

>> Foreign :- Contain other partical then wheat grains

>> Broken_grain :- Contain broken wheat grains.

>> Grain_cover :- Contains the cover of wheat grains.

The dataset is already preprocessed through programming and manually. These five classes of wheat grains are than used for feature extraction process. The feature extraction process takes the wheat grain as input and return features as output. The outputed features the used by classification to classify the partical into one of the above class.

The classifier is written in python, to run the codes python 2.7 and following packages are to be installed:

> numpy==1.15.4

> opencv==3.4.0.12

> Keras==2.1.2

There are two classifier are provided:

> classifier_2_v2.py : This classifier before classification divide dataset into 2 sets i.e. grain/not_grain, where grain contain 'Grain' class and not_grain contain all other class(Damaged_grain, Foreign, Broken_grain, Grain_cover) given above. 

> classifier_5_v2.py : This classifier before classification divides dataset into 5 sets i.e. the above given classes.

To see classification results run the following:

> $ python classifier_2_v2.py     or

> $ python classifier_5_v2.py

To test quality detection:

> $ python cmd_wheat_quality_detector_v2.py

Output

> Enter the file(wheat image) location to dectect : test.jpg

> Segmentation in process...

>	0 Number of segment rejected out of 204 in L1 segmentation

> Level 2 seg. start 1.50203704834e-05

> 2nd Level of segmentation Finished 61.4671521187

> 	In level 2 segmentation 24 rejected

> 	Total number of segments 257

> 	Number of rejected segments 24

> Segmentation in Complete.

> Feature extraction in process...

> Featur extraction in complete.

> Key Error

> Number of good grain : 183

> Number Not good grain or imputity: 49
