from sklearn import preprocessing
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
import PIL
import cv2

weights = EfficientNet_B6_Weights.IMAGENET1K_V1
model = efficientnet_b6(weights=weights)

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
    






