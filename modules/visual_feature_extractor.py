import imp
import os
import cv2
import pdb
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.functional import InterpolationMode
import torchvision.models as models
import logging
import datetime
import sys
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
import cv2
import PIL

weights = SqueezeNet1_1_Weights.IMAGENET1K_V1
model = squeezenet1_1(weights=weights)
# my_model = nn.Sequential(*list(model.children())[:])

my_model = model

# arch = list(model.children())[:-1][0]
# classifier_2d = list(model.children())[-1][:-2]
# classifier_2d

# modules = [layer for layer in arch]
# # modules += [layer for layer in classifier_2d]
# # modules.append(nn.AdaptiveAvgPool2d(output_size=(13, 1)))
# my_model = nn.Sequential(*modules)
# my_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
my_model.to(device)

class VisualFeatures():
    def __init__(self, weights, model, device):
        self.weights = weights
        # remove the classification head
        self.model = model
        self.device = device

    def frame_preprocess(self, img):
        """
        In:
        - img (numpy.ndarray): input image of shape [height, width, num_channels]
        Out:
        - (torch.Tensor): output Tensor of shape [num_channels, height, width]
        """

        img = PIL.Image.fromarray(img)
        preprocess = self.weights.transforms()
        img = preprocess.forward(img)

        return img


    def extract_video_features(file_path, self.model, self.device, flat_ft_size=25, pad_dims = [3, 224, 224]):
        cap = cv2.VideoCapture(file_path)
        batch = []

        video_vf = torch.Tensor().to(device)


        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret:
                batch.append(self.frame_preprocess(frame))
        
            if len(batch) == flat_ft_size:
                batch = torch.stack(batch)
                batch = batch.to(device)
                flat_vf = model(batch)
                # flat_vf, _ = torch.max(flat_vf, dim=0)
                flat_vf = flat_vf.reshape([1]+list(flat_vf.shape))
                video_vf = torch.cat((video_vf, flat_vf))

                batch = []
                
            if not ret:

                if len(batch) != flat_ft_size and batch:
                    batch = torch.stack(batch)
                    padding = torch.zeros([flat_ft_size - len(batch)] + pad_dims)            
                    
                    batch = torch.vstack((batch, padding))
                    batch = batch.to(device)
                    flat_vf = model(batch)
                    # flat_vf, _ = torch.max(flat_vf, dim=0)
                    flat_vf = flat_vf.reshape([1]+list(flat_vf.shape))
        
                    video_vf = torch.cat((video_vf, flat_vf))

                break

        return video_vf
    






