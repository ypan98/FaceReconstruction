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

IS_IMG = True
SHOW_LM = False


sampleName = "sample5"
itemPath = "../data/samples/rgb/" + sampleName + ".jpeg"
landmarkOutputPath = "../data/samples/landmark/" + sampleName + ".txt"


# load model
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector='sfd')

# detect landmarks
if IS_IMG:
    img = cv2.imread(itemPath)
    landmarks = model.get_landmarks_from_image(img)
else:
    video = cv2.VideoCapture(itemPath)
    landmarks = model.get_landmarks_from_batch(video)

# write to file
landmarks = np.array(landmarks)
if IS_IMG:
    landmarks = landmarks.squeeze()
np.savetxt(landmarkOutputPath, landmarks)

if SHOW_LM:
    landmarks = landmarks.astype(int)
    for pixel in landmarks:
        image = cv2.circle(img, (pixel[0],pixel[1]), radius=2, color=(255, 0, 0), thickness=-1)
    cv2.imshow("asd", img)
    cv2.waitKey(0) 
