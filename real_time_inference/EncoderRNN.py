import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import squeezenet1_1
from torchvision.models.feature_extraction import create_feature_extractor

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device, biDirectional = False):
        super(EncoderRNN, self).__init__()
        
        model = squeezenet1_1(pretrained=True).to(config.device)
        return_nodes = {
            'features.12.cat': 'layer12'
        }
        self.pretrained_model = create_feature_extractor(model, return_nodes=return_nodes).to(config.device)
        self.pretrained_model.eval()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.D = 2 if biDirectional else 1

        self.rnn = nn.LSTM(
                input_size = self.input_size,
                hidden_size = self.hidden_size*self.D,
                num_layers = 1,
                dropout = 0,
                bidirectional = biDirectional,
                batch_first = True).to(config.device)

    def forward(self, input, hidden):
        features = self.pretrained_model(input.squeeze())['layer12'].to(device=self.device)
        feat_shape = features.shape

        feat_flat =  torch.reshape(features,(1,feat_shape[0],feat_shape[1]*feat_shape[2]*feat_shape[3])).to(device=self.device)

        output, hidden = self.rnn(feat_flat, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(self.D, 1, self.hidden_size*self.D, device=self.device),
                torch.zeros(self.D, 1, self.hidden_size*self.D, device=self.device))