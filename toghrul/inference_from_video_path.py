import os
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.functional import InterpolationMode
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
import torchviz
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from torchvision import transforms

from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.metrics import classification_report
import torchmetrics

from pytorchvideo.data import labeled_video_dataset

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
    Permute,   
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomAdjustSharpness,
    Resize,
    RandomHorizontalFlip
)

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)


video_transform = Compose([
    ApplyTransformToKey(key="video",
    transform=Compose([
        UniformTemporalSubsample(25),
        Lambda(lambda x: x/255),
        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        # RandomShortSideScale(min_size=256, max_size=512),
        CenterCropVideo(256),
        RandomHorizontalFlip(p=0.5),
    ]),
    ),
])

class VideoModel(LightningModule):
    def __init__(self):
        super(VideoModel, self).__init__()
        
        self.video_model = torch.hub.load('facebookresearch/pytorchvideo', 'efficient_x3d_xs', pretrained=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(400, 1)

        self.lr = 1e-3
        self.batch_size = 8
        self.num_worker = 4
        self.num_steps_train = 0
        self.num_steps_val = 0

        # self.metric = torchmetrics.classification.MultilabelAccuracy(num_labels=num_classes)
        self.metric = torchmetrics.Accuracy()
        
        #loss
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.video_model(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters(), lr = self.lr)
        scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.05, patience=2, min_lr=1e-6)
        # scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=1e-6, last_epoch=-1)
        return {'optimizer': opt,
                'lr_scheduler': scheduler, 
                "monitor": "val_loss"}




def word_level_prediction(path_to_model, path_to_video):
    
    model = VideoModel()
    model.load_state_dict(torch.load(path_to_model))
    
    # model = VideoModel.load_from_checkpoint(
    # checkpoint_path="/home/toghrul/SLR/sign-lang/checkpoints/last-v4.ckpt",
    # hparams_file="/home/toghrul/SLR/sign-lang/lightning_logs/version_45/hparams.yaml",
    # map_location=None,
    # )
    
    model = model.cuda()
    
    video = EncodedVideo.from_path(path_to_video)

    video_data = video.get_clip(0, 2)
    video_data = video_transform(video_data)

    inputs = video_data["video"].cuda()
    inputs = inputs.unsqueeze(0)
    
    preds = model(inputs).detach().cpu().numpy()
    preds = np.where(preds > 0, 1, 0)
    
    return preds