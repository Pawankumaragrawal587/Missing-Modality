import torch.nn as nn
from collections import OrderedDict
import torch

class InferNetNew(nn.Module):

    def __init__(self, mode, output_layers = ['default']):
        super(InferNetNew, self).__init__()
        self.output_layers = output_layers

        self.l1_a = nn.Linear(1024, 128) # to regularize image feature when sound is missing
        self.l2_a = nn.Linear(128, 8) # to regularize image feature when sound is missing

        self.l1_m = nn.Linear(1220, 512) # to regularize image feature when sound is missing
        self.l2_m = nn.Linear(512, 250) # to regularize image feature when sound is missing

        self.l1_v = nn.Linear(1000, 128) # to regularize image feature when sound is missing
        self.l2_v = nn.Linear(128, 32) # to regularize image feature when sound is missing

        num_present = 0
        if (mode & (1<<0)):
            num_present += 1
        if (mode & (1<<1)):
            num_present += 1
        if (mode & (1<<2)):
            num_present += 1

        present_feat_dim = 250 * num_present

        self.fc0_a = nn.Linear(present_feat_dim, 10) # image to sound's weight
        self.fc0_m = nn.Linear(present_feat_dim, 10) # image to sound's weight
        self.fc0_v = nn.Linear(present_feat_dim, 10) # image to sound's weight

        self.fc1 = nn.Linear(present_feat_dim, 750) # image to complete feature (for regularization)
        self.fc11 = nn.Linear(present_feat_dim, 750) # image to complete feature (for regularization)

        self.fc2 = nn.Linear(750, 512) # complete feature (reduced to 256 dim) (for regulariztion)
        self.fc21 = nn.Linear(750, 512) # complete feature (reduced to 256 dim) (for regulariztion)

    def forward(self, x, meta_train=True):

        outputs = OrderedDict()
        if meta_train:
            # print('encoder meta_train=True')
            if 'l1_a' in x.keys():
                mu1 = self.l1_a(x['l1_a'])
                # std1 = torch.exp(0.5*mu1)
                
                outputs['l1_a'] = torch.randn_like(mu1) + mu1

            if 'l1_m' in x.keys():
                mu1 = self.l1_m(x['l1_m'])
                # std1 = torch.exp(0.5*mu1)
                
                outputs['l1_m'] = torch.randn_like(mu1) + mu1

            if 'l1_v' in x.keys():
                mu1 = self.l1_v(x['l1_v'])
                # std1 = torch.exp(0.5*mu1)
                
                outputs['l1_v'] = torch.randn_like(mu1) + mu1

            if 'l2_a' in x.keys():
                mu2 = self.l2_a(x['l2_a'])
                # std2 = torch.exp(mu2)
                
                outputs['l2_a'] = torch.randn_like(mu2) + mu2

            if 'l2_m' in x.keys():
                mu2 = self.l2_m(x['l2_m'])
                # std2 = torch.exp(mu2)
                
                outputs['l2_m'] = torch.randn_like(mu2) + mu2

            if 'l2_v' in x.keys():
                mu2 = self.l2_v(x['l2_v'])
                # std2 = torch.exp(mu2)
                
                outputs['l2_v'] = torch.randn_like(mu2) + mu2

            if 'fc0_a' in x.keys():
                mu0 = self.fc0_a(x['fc0_a'])
                outputs['fc0_a'] = mu0

            if 'fc0_m' in x.keys():
                mu0 = self.fc0_m(x['fc0_m'])
                outputs['fc0_m'] = mu0

            if 'fc0_v' in x.keys():
                mu0 = self.fc0_v(x['fc0_v'])
                outputs['fc0_v'] = mu0

            if 'fc1' in x.keys():
                mu3 = self.fc1(x['fc1'])
                sigma3 = self.fc11(x['fc1'])
                std3 = torch.exp(0.5*sigma3)
                std3 = torch.clamp(std3, min=0, max=1)
                
                outputs['fc1'] = torch.randn_like(mu3) + mu3

            if 'fc2' in x.keys():
                mu4 = self.fc2(x['fc2'])
                sigma4 = self.fc21(x['fc2'])
                std4 = torch.exp(0.5*sigma4)
                std4 = torch.clamp(std4, min=0, max=1)
                
                outputs['fc2'] = torch.randn_like(mu4) + mu4

            return outputs

        else:
            # print('encoder meta_train=False')
            if 'l1_a' in x.keys():
                mu1 = self.l1_a(x['l1_a'])
                # std1 = torch.exp(0.5*mu1)
                
                outputs['l1_a'] = mu1

            if 'l1_m' in x.keys():
                mu1 = self.l1_m(x['l1_m'])
                # std1 = torch.exp(0.5*mu1)
                
                outputs['l1_m'] = mu1

            if 'l1_v' in x.keys():
                mu1 = self.l1_v(x['l1_v'])
                # std1 = torch.exp(0.5*mu1)
                
                outputs['l1_v'] = mu1

            if 'l2_a' in x.keys():
                mu2 = self.l2_a(x['l2_a'])
                # std2 = torch.exp(mu2)
                
                outputs['l2_a'] = mu2

            if 'l2_m' in x.keys():
                mu2 = self.l2_m(x['l2_m'])
                # std2 = torch.exp(mu2)
                
                outputs['l2_m'] = mu2

            if 'l2_v' in x.keys():
                mu2 = self.l2_v(x['l2_v'])
                # std2 = torch.exp(mu2)
                
                outputs['l2_v'] = mu2

            if 'fc0_a' in x.keys():
                mu0 = self.fc0_a(x['fc0_a'])
                outputs['fc0_a'] = mu0

            if 'fc0_m' in x.keys():
                mu0 = self.fc0_m(x['fc0_m'])
                outputs['fc0_m'] = mu0

            if 'fc0_v' in x.keys():
                mu0 = self.fc0_v(x['fc0_v'])
                outputs['fc0_v'] = mu0

            if 'fc1' in x.keys():
                mu3 = self.fc1(x['fc1']) 
                outputs['fc1'] = mu3

            if 'fc2' in x.keys():
                mu4 = self.fc2(x['fc2'])
                outputs['fc2'] = mu4

            return outputs
