import numpy as np

import torch
from pytorchvideo.data.encoded_video import EncodedVideo

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

def word_level_prediction(path_to_model, path_to_video):
    model = torch.load(path_to_model)
    video = EncodedVideo.from_path(path_to_video)

    video_data = video.get_clip(0, 2)
    video_data = video_transform(video_data)

    inputs = video_data["video"].cuda()
    inputs = inputs.unsqueeze(0)
    
    preds = model(inputs).detach().cpu().numpy()
    preds = np.where(preds > 0, 1, 0)
    
    return preds

