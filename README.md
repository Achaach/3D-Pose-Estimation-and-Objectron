# 3D-Pose-Estimation-and-Objectron

## Introduction
Objectron is a 3D object detection and tracking tool developed by Google. It allows for the detection and tracking of 3D objects in real-time using a single RGB camera. Objectron uses machine learning algorithms to generate a 3D bounding box around the detected object and estimate its 3D pose.

This project aims to develop a robust and efficient 3D object detection and tracking system using 3D pose estimation and Objectron techniques. The system can be used in various applications such as robotics, self-driving cars, and augmented reality.
 
 ## Outline
1. 3D bounding box detection on 2D images
2. Estimation of world coordinates of the camera
3. Human Pose Detection
4. Projection of 2D pose estimation to 3D world coordinates
5. Intersection between Pose and Objectron

## Result
<img width="405" alt="objectron" src="https://user-images.githubusercontent.com/90078254/222274775-1e44c1ff-5061-457d-8b63-68a707ba22dd.png">

## MediaPipe Object Detection
In the following parts of project, we perform a use of how MediaPipe works with our object detection resposnibilities, please check the `.ipynb` files in the index for more information.

### Face
In this part we perform a face detection by Mediapipe, which is a widely used application for object detection fields. Please refer the file `Face.ipynb` for more information.

<img width="334" alt="Screen Shot 2023-03-06 at 8 15 26 PM" src="https://user-images.githubusercontent.com/90078254/223294642-fdc92f33-ca47-4465-8b44-14c2dbd6bd75.png">

### Pose Estimation
Human posture estimation is essential for a number of applications, including measuring physical activity, understanding sign language, and controlling full-body gestures. It can serve as the foundation for exercises. Also, it can make it possible for augmented reality to layer digital information and material on top of the real environment. Please refer the `Pose.ipynb` file.

<img width="441" alt="Screen Shot 2023-03-06 at 8 23 52 PM" src="https://user-images.githubusercontent.com/90078254/223297903-7ce46bf5-206e-424a-96c8-654cdbe55b62.png">


### Objectron
As we have introduced in the main file, MediaPipe Objectron is a mobile real-time 3D object detection solution for everyday objects. It detects objects in 2D images, and estimates their poses through a machine learning (ML) model. By using Objectron, we can easily make 3D object detection. Please refer the `Objectron.ipynb` file.

https://user-images.githubusercontent.com/90078254/223296843-34946fbe-544e-48d7-a309-e1a33c073447.mov


### Segmentation
One  broad application of object detection is segmentation. By using MediaPipe, we can do a Selfie Segmentation, the important people in the scene are divided up. Both PCs and smartphones are capable of running it in real-time. The targeted use cases involve video conferencing and selfie effects when the subject is close to the camera (less than two meters). Please refer the `Segmentation.ipynb` file.

<img width="432" alt="Screen Shot 2023-03-06 at 8 46 03 PM" src="https://user-images.githubusercontent.com/90078254/223297845-1484fa0a-48d7-46f6-8df3-d0e5545dee20.png">
