import imp
import pdb
import copy
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