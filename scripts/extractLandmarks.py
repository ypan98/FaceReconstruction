"""
This script is used to extract facial landmarks from an image or video and save the results to a file (precomputing).
The model used is a combination of:
    - S3FD to detect the bounding box of the face.
    - 2D-FAN (Bulat et al.) to produce 68 landmarks.
"""
import argparse
from os import walk
import face_alignment
import cv2
import numpy as np
import torch

IS_IMG = True


parser = argparse.ArgumentParser(description='Computes the landmarks for all images in the specified folder and stores the output in /data/samples/landmark/')
parser.add_argument('-f','--folder', help='Folder where the input images are stored', required=True)
parser.add_argument('-o','--output', help='Output directory path', required=True)
parser.add_argument('-b','--batch', help='Batch size', default=16)
parser.add_argument('--print', default=False, action='store_true')
args = vars(parser.parse_args())

showLm = bool(args["print"])
input_folder = args["folder"]
if not input_folder.endswith('/'):
    input_folder += "/"
files_in_folder = next(walk(input_folder), (None, None, []))[2]
output_folder = args["output"]
if not output_folder.endswith('/'):
    output_folder += "/"
batch_size = int(args["batch"])

# load model
model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device="cuda" if torch.cuda.is_available() else "cpu", face_detector='sfd')

# loop over files
images = [] # for batching
image_names = []
for file in files_in_folder:
    image = cv2.imread(input_folder+file)
    images.append(image)
    image_names.append(file)
    if len(images) == batch_size:
        batch = np.stack(images)
        batch = batch.transpose(0, 3, 1, 2)
        batch = torch.Tensor(batch[:batch_size])
        preds = model.get_landmarks_from_batch(batch)
        # writing out batch result
        for i, pred in enumerate(preds):
            landmarks = np.array(pred)
            landmarks = landmarks.squeeze()
            np.savetxt(output_folder+image_names[i].replace(".png", ".txt"), landmarks)
            # showing result of batch's first item
            if i==0 and showLm:
                landmarks = landmarks.astype(int)
                for pixel in landmarks:
                    image = cv2.circle(images[i], (pixel[0],pixel[1]), radius=2, color=(255, 0, 0), thickness=-1)
                cv2.imshow("Landmark prediction", images[i])
                cv2.waitKey(0)
        # empty current batch images
        images = []
        image_names = []
