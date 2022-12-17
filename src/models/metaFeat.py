import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

class MetaFeat(nn.Module):

    def __init__(self, output_layers = ['default']):
        super(MetaFeat, self).__init__()
        self.output_layers = output_layers

        self.l1 = nn.Linear(1220, 512)
        self.l2 = nn.Linear(512, 250)

        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()

    def forward(self, x, encoder=None, noise=True, meta_train=True, noise_layer=['conv1','conv2']):
        if noise:
            assert encoder is not None

            l1_m = {'l1_m':x}
            x = self.l1(x)
            if 'conv1' in noise_layer:
                x = self.softplus(encoder(l1_m, meta_train=meta_train)['l1_m']) + x
            x = self.relu(x)
            
            l2_m = {'l2_m':x}
            x = self.l2(x)
            if 'conv2' in noise_layer:
                x = self.softplus(encoder(l2_m, meta_train=meta_train)['l2_m']) + x
            x = self.relu(x)

        else:
            x = self.l1(x)
            x = self.relu(x)
            x = self.l2(x)
            x = self.relu(x)

        x = x.view(x.size(0), -1)
        f = x

        return f 
