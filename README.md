## INSTRUCTIONS

Yale Database details: The Yale Face Database contains 165 grayscale images in JPG format of
15 individuals. There are 11 images per subject, one per different facial expression or
configuration: center-light, w/glasses, happy, left-light, w/no glasses, normal, right-light, sad,
sleepy, surprised, and wink.

At different stages, the project had been tested with 15 images (normal image of every subject)
in the test database. For the final version of this project, we have split the database into test and
train database where there are 105 images in training set and 60 images in the test set after
observing the results that are mentioned in the Result and Analysis section of this report.

#### The face recognition module (init.py)
How to run:
1. Simply run the file init.py module.
2. After that, you will be asked to type an eigenvalue (default is 7) and a threshold value (default
is 7500000) .
3. The program is divided into two parts the first part tests the program on real-time video
capture and the second part test the system using the predefined images present in the test
directory.
The module requires the following directories: preprocess, test, train. ​ Preprocess​​ directory stores
the temporary file required for real-time face recognition. ​ Test​​ directory stores the testing
dataset. ​ Train​​ directory stores the training dataset.
The Real-time face recognition part requires access to a web camera on the device. The code
takes a picture from the webcam and runs the algorithm on it. While the other module simply
requires some test images in the test/ directory.

#### Real time face training module (realtime_face.py)
How to run:
1. Simply run the realtime_face.py file.
2. The program requires one input from the user which is the name of the user whose face is
being trained.
3. After that, a web camera stream will start and the user can save the images by pressing ‘s’
when the window is open. The user can stop the stream by pressing ‘q’ while the window is
open.
The real-time face training module takes the input from the webcam and saves the image into
raw_images folder. Then preprocessing is done which detects the face and then resizes the image
into the specified width and height. After preprocessing, the image is saved in the train directory.


#### Crop face for training module (save_to_training.py)
How to run:
1. Run save_to_training.py. The module takes three inputs the input folder name, the cascade file
name, and the output folder name.
2. This module simply takes image directory as input and gives cropped face images as output
which is stored in the train/ directory.
The ​ input folder name​​ is the name of the directory which has various images which need to be
processed. The ​ Cascade file name​​ is the required file for the face detection algorithm, this is the
xml file which is pre-trained to detect faces in images. The ​ output folder name​​ is the name of
the directory which saves the processed images after a face has been successfully been detected.
After a face has been detected, preprocessing is done and the image is resized to the given width
and height specified in the program.
