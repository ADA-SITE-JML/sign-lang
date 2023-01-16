import imp
import pdb
import copy
from turtle import forward
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F

class TemporalConv(nn.Module):

    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type


        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []

        for layer_id, ks in enumerate(self.kernel_size):
            input_dims = self.input_size if layer_id == 0 else self.hidden_size
            
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(nn.Conv1d(input_dims, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0))

                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))

        self.temporal_conv = nn.Sequential(*modules)

        if self.num_classes != 1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    
    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.div(feat_len, 2)
            else:
                feat_len -= int(ks[1]) - 1
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_ft = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)

        logits = None if self.num_classes == -1 else self.fc(visual_ft.transpose(1, 2)).transpose(1, 2)

        return {
            "visual_feat": visual_ft.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1),
            "feat_len": lgt.cpu()
        }
            
