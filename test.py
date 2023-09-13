from fastapi import FastAPI, UploadFile, File
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps    
import os
from  torchvision import *
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn import metrics
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import os

# Initialize FastAPI app
app = FastAPI()

# Define the Siamese Network class and load the pre-trained model
#create the Siamese Neural Network
class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 128, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(128,256,kernel_size=4),
            nn.ReLU(inplace=True),
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(6400, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,2)
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.cnn1(x)
        #print(output.shape)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        euclidean_distance = F.pairwise_distance(output1, output2)
        return euclidean_distance
net1 = SiameseNetwork()
muzzle_points = torch.load('/Users/rishabh/Documents/work/cattle_project/backend/muzzle_model_with_angle_100_epochs.pth', map_location=torch.device('cpu'))
net1.load_state_dict(muzzle_points)

# Define the image transformation
transformation = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])

# Define a function to preprocess an image
def preprocess_img(file):
    img = Image.open(file)
    img = img.convert("RGB")
    img = transformation(img)
    img = img[None, :, :, :]
    return img

# Define an API endpoint to compare two images
@app.post("/compare_images/")
async def compare_images(file1: UploadFile, file2: UploadFile):
    try:
        # Preprocess the uploaded images
        image1 = preprocess_img(file1.file)
        image2 = preprocess_img(file2.file)

        # Calculate the similarity using the Siamese Network
        with torch.no_grad():
            similarity_score = net1(image1, image2).item()

        return JSONResponse(content={"similarity_score": int(similarity_score)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
