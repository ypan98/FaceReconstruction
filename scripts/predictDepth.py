import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt

filename = "sampleHighRes"
inputFilename = "../data/samples/rgb/" + filename + ".jpeg"
outputFilename = "../data/samples/depth/" +  filename + ".jpeg"

model_type = "DPT_Large" 
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


img = cv2.imread(inputFilename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img).to(device)
with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()   
image = cv2.normalize(output, output, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite(outputFilename, image)