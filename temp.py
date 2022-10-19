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
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import PIL
from typing import Optional, Tuple



def img_resize(img_path, dims=(256, 256)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=dims)

    return img

def frame_preprocess(img):
    """
    In:
    - img (numpy.ndarray): input image of shape [height, width, num_channels]
    Out:
    - (torch.Tensor): output Tensor of shape [num_channels, height, width]
    """

    img = PIL.Image.fromarray(img)
    preprocess = weights.transforms()
    img = preprocess.forward(img)

    return img


def extract_video_features(file_path, model, flat_ft_size=25, pad_dims = [3, 224, 224]):
    cap = cv2.VideoCapture(file_path)
    batch = []
    # ret = True
    count = 0

    video_vf = torch.Tensor()


    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            batch.append(frame_preprocess(frame))

        count += 1
    
        if count % flat_ft_size == 0:
            batch = torch.stack(batch)
            flat_vf = model(batch)
            video_vf = torch.cat((video_vf, flat_vf))

            batch = []
            
        if not ret:
            if len(batch) != flat_ft_size:
                batch = torch.stack(batch)
                padding = torch.zeros([flat_ft_size - len(batch)] + pad_dims)            
                
                batch = torch.vstack((batch, padding))
                flat_vf = model(batch)
                video_vf = torch.cat((video_vf, flat_vf))

            break

    return video_vf


weights = SqueezeNet1_1_Weights.IMAGENET1K_V1
model = squeezenet1_1(weights=weights)

my_model = nn.Sequential(*list(model.children())[:])

preprocess = weights.transforms()

gloss_df_path = "data_validation/processed_gloss.csv"
gloss_df = pd.read_csv(gloss_df_path)
gloss_df.dropna(inplace=True)

read_src = "../data/gloss"
write_src = "../data/visual-features"
log_file_path = "../data/logs/visual-features"


if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)

if not os.path.exists(write_src):
    os.mkdir(write_src)

total_vf = []
d = datetime.datetime.now()
log_filename = os.path.join(log_file_path, f"logfile_{d.strftime('%Y-%m-%d_%H-%M-%S')}.log")
print(log_filename)
logging.basicConfig(filename = log_filename,
                    filemode = "w+",
                    level = logging.DEBUG)

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

logging.info(f"Logging Session Started at {datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}")

error = False

for gloss, df in gloss_df.groupby(by='gloss'):

    gloss_vf = []
    gloss_vf_path = os.path.join(write_src, gloss)

    if not os.path.exists(gloss_vf_path):
        os.mkdir(gloss_vf_path)
        

    gloss_vf_csv_path = os.path.join(gloss_vf_path, f"{gloss}_vf.csv")

    for i in range(len(df)):
        
        df.reset_index(drop=True, inplace=True)
        video_path = os.path.join(read_src, gloss, df.loc[i, 'fileName'])

        if os.path.exists(video_path):
            try:
                gloss_vf.append(extract_video_features(video_path, my_model))
            except RuntimeError as err:
                logging.error(f">>> Problem during visual feature extraction for {gloss}/{df.loc[i, 'fileName']}. See the error message below:\n{err}")
                error = True
                break

            gloss_vf.append(extract_video_features(video_path, my_model))
            
        logging.info(f">>> Succesful visual feature extraction for {gloss}/{df.loc[i, 'fileName']}")
    if error:
        break

    gloss_vf_df = pd.Series(gloss_vf)
    gloss_vf_df.to_csv(gloss_vf_csv_path, index=False)

    total_vf.append([gloss, gloss_vf])

logging.info(f"Logging Session Ended at {datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}")


