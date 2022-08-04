"""
Script used to preprocess the RGBD video sequence recorded with Kinect in order to use it for our face reconstruction program.
Center cropping, switching black <-> white, background to black. For RGB, only cropping is performed
"""
import argparse
import cv2
import numpy as np
from os import walk

parser = argparse.ArgumentParser(description='Preprocess the image sequence recorded by Kinect')
parser.add_argument('-f','--folder', help='Folder relative path', required=True)
parser.add_argument('-m','--mode', help='Mode: either rgb or depth', required=True)
parser.add_argument('-o','--output', help='Output directory path', required=True)
parser.add_argument('-r','--range', help='Max wanted depth range (meters)', default=0.75)
parser.add_argument('-wh','--height', help='Wanted croped image height', default=480)
parser.add_argument('-ww','--width', help='Wanted croped image width', default=640)
args = vars(parser.parse_args())

mode = args["mode"]
input_folder = args["folder"]
if not input_folder.endswith('/'):
    input_folder += "/"
files_in_folder = next(walk(input_folder), (None, None, []))[2]
output_folder = args["output"]
if not output_folder.endswith('/'):
    output_folder += "/"
output_h = int(args["height"])
output_w = int(args["width"])
max_depth = args["range"]


# center crop with wanted height and width
def center_crop(img, wanted_h, wanted_w):
    shape = img.shape
    start_h = 0
    start_w = 0
    if shape[0] > wanted_h:
       start_h = int((shape[0]-wanted_h)/2) 
    if shape[1] > wanted_w:
       start_w = int((shape[1]-wanted_w)/2) 
    end_h = start_h + wanted_h
    end_w = start_w + wanted_w
    return img[start_h:end_h, start_w:end_w]

# preprocess depth map by transforming from cm to m, removing background (> max_depth), flipping (whiter = higher distance) and normalizing to [0, 255]
def preprocess_depth_map(image):
    # background removal (to 0)
    image[np.where(image > max_depth * 1000)] = 0
    # min of detected points
    # minimun = np.min(image[np.where(image != 0)])
    # range of the remaining points
    # range = np.max(image)-minimun
    # to [0, 1]
    # image[np.where(image != 0)] -= minimun
    # image[np.where(image != 0)] /= range
    # flipping (only point where is not 0)
    # image[np.where(image != 0)] = 1-image[np.where(image != 0)]
    # to [0, 255]
    # image = (255*image).astype(int)
    return image

# loop over files
for file in files_in_folder:
    image = cv2.imread(input_folder+file, cv2.IMREAD_UNCHANGED)
    image = center_crop(image, output_h, output_w)
    if mode == "depth":
        image = preprocess_depth_map(image)
    cv2.imwrite(output_folder+"Y"+file, image)
