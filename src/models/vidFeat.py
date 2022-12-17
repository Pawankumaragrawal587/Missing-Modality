import torch.nn as nn

class VidFeat(nn.Module):

    def __init__(self, ):
        super(VidFeat, self).__init__()

        self.l1 = nn.Linear(1000, 128)
        self.l2 = nn.Linear(128, 32)
        self.l3 = nn.Linear(30*32, 250)

        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()

    def forward(self, x, encoder=None, noise=True, meta_train=True, noise_layer=['conv1','conv2']):
        if noise:
            assert encoder is not None

            l1_v = {'l1_v':x}
            x = self.l1(x)
            if 'conv1' in noise_layer:
                x = self.softplus(encoder(l1_v, meta_train=meta_train)['l1_v']) + x
            x = self.relu(x)
            
            l2_a = {'l2_v':x}
            x = self.l2(x)
            if 'conv2' in noise_layer:
                x = self.softplus(encoder(l2_v, meta_train=meta_train)['l2_v']) + x
            x = self.relu(x)
       	else:
            x = self.l1(x)
            x = self.relu(x)
            x = self.l2(x)
            x = self.relu(x)

        x = x.view(x.size(0), -1)
        x = self.l3(x)
        x = self.relu(x)
        f = x

        return f
