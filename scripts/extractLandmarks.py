"""
This script is used to extract facial landmarks from an image or video and save the results to a file (precomputing).
The model used is a combination of:
    - S3FD to detect the bounding box of the face.
    - 2D-FAN (Bulat et al.) to produce 68 landmarks.
"""
import argparse
import face_alignment
import cv2
import numpy as np
import torch

IS_IMG = True


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-f','--file', help='Image name (this file should be placed inside /data/samples/rgb/)', required=True)
parser.add_argument('--print', default=False, action='store_true')
args = vars(parser.parse_args())
sampleName = args["file"]
showLm = args["print"]

itemPath = "../data/samples/rgb/" + sampleName + ".png"
landmarkOutputPath = "../data/samples/landmark/" + sampleName + ".txt"


# load model
model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device="cuda" if torch.cuda.is_available() else "cpu", face_detector='sfd')

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
print("Landmarks saved at ", landmarkOutputPath)

if showLm:
    landmarks = landmarks.astype(int)
    for pixel in landmarks:
        image = cv2.circle(img, (pixel[0],pixel[1]), radius=2, color=(255, 0, 0), thickness=-1)
    cv2.imshow("asd", img)
    cv2.waitKey(0) 
