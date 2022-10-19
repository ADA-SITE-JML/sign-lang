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
my_model = nn.Sequential(*list(model.children())[:])

class VisualFeatures():
    def __init__(self, weights, model):
        self.weights = weights
        # remove the classification head
        self.model = nn.Sequential(*list(model.children())[:-1])

    def preprocess(self, img):
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

    def flattened_model_vf(self, file_path):
        """
        In:
        - img (numpy.ndarray or list of numpy.ndarray): input image batch of shape [batch_size, height, width, num_channels]
        Out:
        - (torch.Tensor): output Tensor of shape [batch_size, num_channels, height, width]
        """

        cap = cv2.VideoCapture(file_path)
        batch = []
        ret = True

        batch = []
        while cap.isOpened() and ret:
            ret, frame = cap.read()
            batch.append(self.preprocess(frame))

        batch = torch.stack(batch)

        flat_vf = self.model(batch)

        return flat_vf
    






