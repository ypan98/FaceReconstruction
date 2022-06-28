"""
This script is used to extract facial landmarks from an image or video and save the results to a file (precomputing).
The model used is a combination of:
    - S3FD to detect the bounding box of the face.
    - 2D-FAN (Bulat et al.) to produce 68 landmarks.
"""

import face_alignment
import cv2
import numpy as np
import torch

isImage = True
sampleName = "sample1"
itemPath = "../samples/rgb/" + sampleName + ".bmp"
landmarkOutputPath = "../samples/landmark/" + sampleName + ".txt"


# load model
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector='sfd')

# detect landmarks
if isImage:
    img = cv2.imread(itemPath)
    landmarks = model.get_landmarks_from_image(img)
else:
    video = cv2.VideoCapture(itemPath)
    landmarks = model.get_landmarks_from_batch(video)

# write to file
landmarks = np.array(landmarks)
if isImage:
    landmarks = landmarks.squeeze()
np.savetxt(landmarkOutputPath, landmarks)
