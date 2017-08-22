# Face Recognition

[![Datmo Model](https://datmo.io/shabazp/datmo-face-recognition/badge.svg)](https://datmo.io/shabazp/datmo-face-recognition)

Computer vision model for facial recognition. This repository comprises main tasks required to be done using facial recognition. For training run, `python recognition.py`

### 1. Face Detection ([detection](https://github.com/Acusense/face-recognition/blob/master/src/detection.py))
* To find bounding box for faces.
* To find number of people in any image.
* Usage: Application can range like Facebook using this to find count of people and detect face to let users tag it.

### 2. Facial Landmark Detection ([landmark_detection](https://github.com/Acusense/face-recognition/blob/master/src/landmark_detection.py))

* To find the face feature locations (eyes, nose, etc) for each face in the image
* Usage: Application can lie to apply makeup application or is the founding basis to overlay structure as Snapchat does for [lenses](https://support.snapchat.com/en-US/article/lenses1)

### 3. Facial Verification ([verification](https://github.com/Acusense/face-recognition/blob/master/src/verification.py))
* Given an image of face, we can compare if any new image contains the same person.
* Usage: Widely used application eg. Uber uses to check if driver is the same person as registered on [app](https://newsroom.uber.com/securityselfies/)

### 4. Facial Recognition ([recognition](https://github.com/Acusense/face-recognition/blob/master/src/recognition.py)/[training](https://github.com/Acusense/face-recognition/blob/master/src/recognition_training.ipynb))

* In a given number of classes does any new image of a face lie in one of the classes.
* Usage: Application lie to identify any celebrety and also used by Facebook to identify friends in moments or by Google in their Photos app.


  Built using [dlib](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html) and [face_recognition](https://github.com/ageitgey/face_recognition)
