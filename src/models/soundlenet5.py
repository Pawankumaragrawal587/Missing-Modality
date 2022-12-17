import torch
import torch.nn as nn

class SoundLenet5New(nn.Module):
    """docstring forLenet5 Sound"""
    def __init__(self, extractor1, extractor2, extractor3):
        super(SoundLenet5New, self).__init__()

        self.audio_extractor = extractor1
        self.meta_extractor = extractor2
        self.video_extractor = extractor3

        self.a1 = nn.Linear(1024, 128)
        self.a2 = nn.Linear(128, 8)
        self.a3 = nn.Linear(300*8, 250)

        self.m1 = nn.Linear(1220, 512)
        self.m2 = nn.Linear(512, 250)

        self.v1 = nn.Linear(1000, 128)
        self.v2 = nn.Linear(128, 32)
        self.v3 = nn.Linear(30*32, 250)

        self.fc_user = nn.Linear(1220, 512)
        self.fc1 = nn.Linear(750, 512)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, label, user_embedding, audio, meta, video, encoder=None, sound_mean=None, noise_layer=['conv1','conv2','fc1','fc2'], meta_train=True, mode=7):

        if mode < 7:
            assert sound_mean is not None
            assert noise_layer is not None
            assert encoder is not None

            x = None

            if (mode & (1<<0)):
                # print('---> ', torch.cuda.memory_allocated())
                audio_feature = self.audio_extractor(audio, encoder=encoder, noise=True, meta_train=meta_train, noise_layer=noise_layer) # image feature: 160 dim vector
                # print('------> ', torch.cuda.memory_allocated())
                if x is None:
                    x = audio_feature
                else:
                    x = torch.cat([x, audio_feature], dim=1)

            if (mode & (1<<1)):
                # print('---> ', torch.cuda.memory_allocated())
                meta_feature = self.meta_extractor(meta, encoder=encoder, noise=True, meta_train=meta_train, noise_layer=noise_layer) # image feature: 160 dim vector
                # print('------> ', torch.cuda.memory_allocated())
                if x is None:
                    x = meta_feature
                else:
                    x = torch.cat([x, meta_feature], dim=1)

            if (mode & (1<<2)):
                # print('---> ', torch.cuda.memory_allocated())
                video_feature = self.video_extractor(video, encoder=encoder, noise=True, meta_train=meta_train, noise_layer=noise_layer) # image feature: 160 dim vector
                # print('------> ', torch.cuda.memory_allocated())
                if x is None:
                    x = video_feature
                else:
                    x = torch.cat([x, video_feature], dim=1)

            if not (mode & (1<<0)):
                audio_mean = sound_mean[0].expand(x.shape[0], -1, -1)

                fc0_a = {'fc0_a': x}
                if 'fc0' in noise_layer: # add noise to the concatenated feature: 480 dimension. 
                    
                    weight = self.softplus(encoder(fc0_a, meta_train=meta_train)['fc0_a']).unsqueeze(-1)

                    audio_feature =  audio_mean.matmul(weight)
                   
                    for i in range(audio_feature.shape[0]):
                        audio_feature[i] = audio_feature[i].clone() / weight.sum(1)[i]

                audio_feature = audio_feature.view(audio_feature.size(0), 300, 1024)
                audio_feature = self.a1(audio_feature)
                audio_feature = self.relu(audio_feature)
                audio_feature = self.a2(audio_feature)
                audio_feature = self.relu(audio_feature)

                audio_feature = audio_feature.view(audio_feature.size(0), -1)

                audio_feature = self.a3(audio_feature)
                audio_feature = self.relu(audio_feature)

            if not (mode & (1<<1)):
                meta_mean = sound_mean[1].expand(x.shape[0], -1, -1)

                fc0_m = {'fc0_m': x}
                if 'fc0' in noise_layer:

                    weight = self.softplus(encoder(fc0_m, meta_train=meta_train)['fc0_m']).unsqueeze(-1)

                    meta_feature =  meta_mean.matmul(weight)

                    for i in range(meta_feature.shape[0]):
                        meta_feature[i] = meta_feature[i].clone() / weight.sum(1)[i]

                meta_feature = meta_feature.view(meta_feature.size(0), -1)
                meta_feature = self.m1(meta_feature)
                meta_feature = self.relu(meta_feature)
                meta_feature = self.m2(meta_feature)
                meta_feature = self.relu(meta_feature)

            if not (mode & (1<<2)):
                video_mean = sound_mean[2].expand(x.shape[0], -1, -1)

                fc0_v = {'fc0_v': x}
                if 'fc0' in noise_layer:
                    
                    weight = self.softplus(encoder(fc0_v, meta_train=meta_train)['fc0_v']).unsqueeze(-1)

                    video_feature =  video_mean.matmul(weight)
                   
                    for i in range(video_feature.shape[0]):
                        video_feature[i] = video_feature[i].clone() / weight.sum(1)[i]

                video_feature = video_feature.view(video_feature.size(0), 30, 1000)
                video_feature = self.v1(video_feature)
                video_feature = self.relu(video_feature)
                video_feature = self.v2(video_feature)
                video_feature = self.relu(video_feature)

                video_feature = video_feature.view(video_feature.size(0), -1)

                video_feature = self.v3(video_feature)
                video_feature = self.relu(video_feature)

            x = x.view(x.size(0), -1)

            fc1 = {'fc1':x}
            
            x = torch.cat([audio_feature, meta_feature, video_feature], dim=1)

            complete_feature = x

            if 'fc1' in noise_layer:
                
                x = self.softplus(encoder(fc1, meta_train=meta_train)['fc1']) + x

            f = x

            fc2 = {'fc2':x}
            x = self.relu(self.fc1(x))
            if 'fc2' in noise_layer:
                x = self.softplus(encoder(fc2, meta_train=meta_train)['fc2']) + x

            f1 = x

            x = self.dropout(x)
            user_embedding = user_embedding.view(user_embedding.size(0), -1)
            user_embedding = self.fc_user(user_embedding)
            x = torch.cat([x, user_embedding], dim=1)

            x = self.fc2(x)
            x = self.relu(x)

            x = self.fc3(x)
            x = self.sigmoid(x)
            return x, f, f1, complete_feature

        elif mode == 7:
            audio_feature = self.audio_extractor(audio, encoder=None, noise=False, meta_train=meta_train, noise_layer=noise_layer) # image feature: 160 dim vector
            meta_feature = self.meta_extractor(meta, encoder=None, noise=False, meta_train=meta_train, noise_layer=noise_layer) # image feature: 160 dim vector
            video_feature = self.video_extractor(video, encoder=None, noise=False, meta_train=meta_train, noise_layer=noise_layer) # image feature: 160 dim vector

            x = torch.cat([audio_feature, meta_feature, video_feature], dim=1)
            f = x

            x = self.relu(self.fc1(x))
            f1 = x
            x = self.dropout(x)

            user_embedding = user_embedding.view(user_embedding.size(0), -1)
            user_embedding = self.fc_user(user_embedding)

            x = torch.cat([x, user_embedding], dim=1)

            x = self.fc2(x)
            x = self.relu(x)

            x = self.fc3(x)
            x = self.sigmoid(x)
            return x, f, f1, f
        else:
            raise ValueError('mode should be one or two.')
