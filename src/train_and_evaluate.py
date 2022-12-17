from __future__ import print_function, absolute_import, division

import os
import time
import math
import datetime
import argparse
import os.path as path
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.vidFeat import VidFeat
from models.metaFeat import MetaFeat
from models.audioFeat import AudioFeat
from models.newencoder import InferNetNew
from models.soundlenet5 import SoundLenet5New
from models.loss import KDFeatureLoss, KDFeatureLossTwo, KDLossAlignTwo

from dataset.meta_training_dataset import MetaTrSouMNIST
from dataset.soundmnist import SoundMNIST

from utils.misc import AverageMeter, AvgF1, save_ckpt, save_ckpt_inferNet, save_ckpt_classifier

from metann import ProtoModule

import gc

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')
    parser.add_argument('-i', '--image_root', default = '../data/mnist/', type = str, help='data root' )
    parser.add_argument('-s', '--sound_root', default = '../data/sound_450/', type = str, help='data root' )
    parser.add_argument('--checkpoint', default='./save/metadrop_new/feature/15', type=str, help='checkpoint directory')
    parser.add_argument('--snapshot', default=1, type=int, help='save models for every # epochs (default: 1)')

    parser.add_argument('--soundmnist_model_path', default='./save/soundmnist/path/', 
                        type=str, help='pre-trained sound mnist model')
    parser.add_argument('--soundmnist_model_name', default='name/of/best_model.path.tar', 
                        type=str, help='pre-trained sound mnist model')


    parser.add_argument('--sound_mean_path', default='./save/sound_mean/kmean/path/', 
                        type=str, help='pre calculated sound mean path')
    parser.add_argument('--sound_mean_name', default='sound_mean_150.npy', 
                        type=str, help='pre calculated sound mean name')
    

    # model related parm
    parser.add_argument('-b', '--batch_size', default = 128, type = int, help='batch size' )
    parser.add_argument('--per_class_num', default = 15 , type = int, help='per_class_num' ) # 15 * 10 = 150 sound data available, total 1500 sound data
    parser.add_argument('--iterations', default = 11000 , type = int, help='num of epoch' )
    parser.add_argument('--lr', default = 1e-3, type = float, help='initial learning rate' )
    parser.add_argument('--lr_inner', default = 1e-3, type = float, help='initial learning rate' )
    parser.add_argument('--inner_loop', default = 1, type = int, help='meta_train inner_loop' )
    parser.add_argument('--mc_size', default = 30, type = int, help='MC size for meta-test' )
    parser.add_argument('--vis_device', default='0', type=str, help='set visiable device')
    parser.add_argument('--num_modality', default=3, type=int, help='Number of Modalities')
    parser.add_argument('--modality_complete_ratio', default=50, type=int, help='Fraction of modality complete samples')
    parser.add_argument('--modality_complete_list', default=[0, 1, 0], type=list, help='Each element of list represents fraction of samples with modality available from modality incomplete samples')

    args = parser.parse_args()

    return args

def main(args):
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.vis_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_modality = args.num_modality
    print('Modality Complete Ratio: ', args.modality_complete_ratio)
    print('Modality Complete list: ', args.modality_complete_list)
    # meta-training dataset
    meta_train_loader = []
    encoder_list = []
    for i in range(1, (2 ** num_modality) -1):
        # meta-train step dataset
        meta_train_dataset = MetaTrSouMNIST(mode=i,
                                            modality_complete_list = args.modality_complete_list,
                                            modality_complete_ratio=args.modality_complete_ratio)
        print('Mode:', i, ', Data Size:', len(meta_train_dataset))
        if len(meta_train_dataset) > 0:
            data_loader = DataLoader(meta_train_dataset, batch_size = args.batch_size, shuffle = True,  
                                       num_workers=0, pin_memory=True)
        else:
            data_loader = None
        meta_train_loader.append(data_loader)
        
        encoder = InferNetNew(i).to(device) # auxilary model to infer the missing modality
        encoder = ProtoModule(encoder)
        encoder_list.append(encoder)

    meta_val_dataset = MetaTrSouMNIST(mode=(2 ** num_modality) - 1,
                                      modality_complete_list = args.modality_complete_list,
                                      modality_complete_ratio=args.modality_complete_ratio)
    meta_val_loader = DataLoader(meta_val_dataset, batch_size = args.batch_size, shuffle = True, 
                                 num_workers=0, pin_memory=True)

    meta_test_dataset = SoundMNIST(img_root=args.image_root,sound_root=args.sound_root, per_class_num=args.per_class_num, train=False)
    meta_test_loader = DataLoader(meta_test_dataset, batch_size = args.batch_size, shuffle = False, num_workers=0, pin_memory=True)
    print('val data size:', len(meta_val_dataset))
    print('test data size:',len(meta_test_dataset))


    # create model 
    print('==> model creating....')

    a_extractor = AudioFeat().to(device) # audio extractor
    m_extractor = MetaFeat().to(device) # meta extractor
    v_extractor = VidFeat().to(device) # video extractor

    image_sound_extractor = SoundLenet5New(a_extractor, m_extractor, v_extractor).to(device) # multimodal fusion model

    # ckpt_image_sound = torch.load(path.join(args.soundmnist_model_path, args.soundmnist_model_name)) # load pre-trained weight
    # image_sound_extractor.load_state_dict(ckpt_image_sound['state_dict'])

    image_sound_extractor = ProtoModule(image_sound_extractor)

    # load pre calculated sound mean
    name = 'modality_prior_' + str(args.modality_complete_ratio) + '_' + str(args.modality_complete_list[0]) + '_' + str(args.modality_complete_list[1]) + '_' + str(args.modality_complete_list[2])
    audio_mean = np.load(os.path.join(args.sound_mean_path, name + '_0.npy'))
    audio_mean = torch.from_numpy(audio_mean).T.to(device)
    meta_mean = np.load(os.path.join(args.sound_mean_path, name + '_1.npy'))
    meta_mean = meta_mean.astype(np.float32)
    meta_mean = torch.from_numpy(meta_mean).T.to(device)
    video_mean = np.load(os.path.join(args.sound_mean_path, name + '_2.npy'))
    video_mean = torch.from_numpy(video_mean).T.to(device)

    sound_mean = [audio_mean, meta_mean, video_mean]
    
    print('==> model has been created')
    print("==> Total parameters (reference): {:.2f}M".format(sum(p.numel() for p in image_sound_extractor.parameters()) / 1000000.0))
    
    
    criterion_meta_train = nn.MSELoss().to(device)
    criterion_meta_val = KDLossAlignTwo(alpha = 0.01, beta = 0.01).to(device)

    optimizer_image_sound = torch.optim.Adam(image_sound_extractor.parameters(), lr = args.lr, weight_decay = 1e-4)
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr = args.lr, weight_decay = 1e-4)

    
    scheduler_image_sound = torch.optim.lr_scheduler.StepLR(optimizer_image_sound, step_size = 5000, gamma = 0.1)  
    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size = 5000, gamma = 0.1)



    best_val_acc =None
    
    glob_step = 0

    evl_step = 0

    ckpt_dir_path = path.join(args.checkpoint, datetime.datetime.now().isoformat())
    # print(ckpt_dir_path)
    if not path.exists(ckpt_dir_path):
            os.makedirs(ckpt_dir_path)
            print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))

    writer = SummaryWriter(log_dir=ckpt_dir_path)

    print('start training')
    for iterate in range(args.iterations):
        if iterate%100 == 0:
            print('Iteration: ', iterate)
        meta_val_batch = next(iter(meta_val_loader))
        for mode in range(1, (2 ** num_modality) -1):

            if meta_train_loader[mode-1] is None:
                continue

            meta_train_batch = next(iter(meta_train_loader[mode-1]))

            glob_step = meta_training(args, meta_train_batch, meta_val_batch, mode, sound_mean, image_sound_extractor, encoder_list[mode-1], criterion_meta_train, criterion_meta_val, optimizer_image_sound, optimizer_encoder, device, writer, iterate, args.iterations)

            scheduler_image_sound.step()
            scheduler_encoder.step()

            if (iterate >= 5500) and ((iterate) % 1100 == 0 or iterate == args.iterations - 1):

                mtr_image_acc = eval(criterion_meta_train, meta_train_loader[mode-1], image_sound_extractor, sound_mean, device, writer, encoder=encoder_list[mode-1], mode=mode)
                torch.set_grad_enabled(True)
                print('Iteration:[{}/{}], mode:{}, Meta-Train RMSE:{:.6f}' .format(iterate, args.iterations, mode, mtr_image_acc))

        if (iterate >= 5500) and ((iterate) % 1100 == 0 or iterate == args.iterations - 1):

            test_image_acc = 0.
            for i in range(5):
                test_image_acc += eval(criterion_meta_train, meta_test_loader, image_sound_extractor, sound_mean, device, writer, encoder=None, mode=7)
            test_image_acc = test_image_acc / 5

            mval_image_acc = eval(criterion_meta_train, meta_val_loader, image_sound_extractor, sound_mean, device, writer, encoder=None, mode=7)
            
            torch.set_grad_enabled(True)
            print('Iteration:[{}/{}], Meta-Val RMSE:{:.6f}, Test RMSE:{:.6f}' .format(iterate, args.iterations, mval_image_acc, test_image_acc))

        # save best model
        if iterate >= 5500:
            
            if best_val_acc is None or best_val_acc > test_image_acc:
                best_val_acc = test_image_acc

                torch.set_grad_enabled(True)
                print('Iteration:[{}/{}], Best Test RMSE:{:.6f} ' .format(iterate, args.iterations, best_val_acc))
                save_ckpt_classifier({'iterate': iterate, 'lr': args.lr, 'state_dict': image_sound_extractor.state_dict(),
                         'optimizer': optimizer_image_sound.state_dict() }, ckpt_dir_path, iteration =  iterate , best_acc = best_val_acc)
                save_ckpt_inferNet({'iterate': iterate, 'lr': args.lr, 'step': glob_step, 'state_dict': encoder.state_dict(), 
                             'optimizer': optimizer_encoder.state_dict() }, ckpt_dir_path, iteration = iterate )
    writer.close()
    

def meta_training(args, meta_train_batch, meta_val_batch, mode, sound_mean, image_sound_extractor, encoder, criterion_meta_train, criterion_meta_val, optimizer_image_sound, optimizer_encoder, device, writer, iterate, total_iterate):
    ''' train one epoch'''

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses= AverageMeter()

    torch.set_grad_enabled(True)

    batch_size = meta_train_batch[0].shape[0]

    # batch of meta train: sampled from image modality
    meta_train_audio = meta_train_batch[0].to(device)
    meta_train_meta = meta_train_batch[1].to(device)
    meta_train_video = meta_train_batch[2].to(device)
    meta_train_user_embedding = meta_train_batch[3].to(device)
    meta_train_label = meta_train_batch[4].to(device)
    meta_train_label = torch.reshape(meta_train_label, (meta_train_label.shape[0], 1))

    # batch of meta validaiton: sampled from both image and sound modality
    meta_val_audio = meta_val_batch[0].to(device)
    meta_val_meta = meta_val_batch[1].to(device)
    meta_val_video = meta_val_batch[2].to(device)
    meta_val_user_embedding = meta_val_batch[3].to(device)
    meta_val_label = meta_val_batch[4].to(device)
    meta_val_label = torch.reshape(meta_val_label, (meta_val_label.shape[0], 1))

    # meta-training 
    params = list(image_sound_extractor.parameters())
    for i in params:
        if not i.requires_grad:
            i.requires_grad = True

    loss_meta_train = 0.
    loss_meta_val = 0.
    mse_loss = nn.MSELoss(reduction='mean')

    for idx in range(args.inner_loop):
        if idx == 0:
            params_star = params
        pred_meta_train_noised,_,_,_ = image_sound_extractor.functional(params_star, True)(meta_train_label, meta_train_user_embedding, meta_train_audio, meta_train_meta, meta_train_video, encoder=encoder,  sound_mean=sound_mean, noise_layer=['fc0','fc1','fc2'], meta_train=True, mode=mode)
        loss_meta_train = torch.sqrt(criterion_meta_train(pred_meta_train_noised, meta_train_label))
        torch.autograd.set_detect_anomaly(True)
        grads = torch.autograd.grad(loss_meta_train, params_star, allow_unused=True, create_graph=True)  # create_graph=True: allow second order derivative
        # print(grads)
        # print(params_star)
        for i in range(len(params_star)):
            # if i <= 7 or i >=26:# not update the sound branch
            if grads[i] is not None: # unused parameters have no gradient 
                params_star[i] = (params_star[i] - args.lr_inner*(0.1**(iterate//1000))*grads[i]).requires_grad_()

    pred_meta_val_noised, f_meta_val_noised1,f_meta_val_noised2, sound_mean_val_noised = image_sound_extractor.functional(params_star, True)(meta_val_label, meta_val_user_embedding, meta_val_audio, meta_val_meta, meta_val_video, encoder=encoder, sound_mean=sound_mean, noise_layer=['fc0','fc1','fc2'], meta_train=False, mode=mode)

    pred_meta_val_clean, f_meta_val_clean1,f_meta_val_clean2, sound_mean_val_clean = image_sound_extractor.functional(params_star, True)(meta_val_label, meta_val_user_embedding, meta_val_audio, meta_val_meta, meta_val_video, mode=7)
    
    sound_mean_val_mse = mse_loss(sound_mean_val_clean, sound_mean_val_noised)

    loss_meta_val = criterion_meta_val(sound_mean_val_clean, sound_mean_val_noised, f_meta_val_clean1, f_meta_val_noised1, f_meta_val_clean2, f_meta_val_noised2, pred_meta_val_noised, pred_meta_val_clean, meta_val_label)

    optimizer_encoder.zero_grad()
    optimizer_image_sound.zero_grad()
    (loss_meta_train + loss_meta_val).backward()
    optimizer_encoder.step()
    optimizer_image_sound.step()
    torch.cuda.empty_cache()

    if (iterate) % 100 == 0:
        print('Iteration [{}/{}], meta-train Loss: {:.4f}, meta-val Loss: {:.4f},' .format(iterate, total_iterate, 
                                                                                       loss_meta_train.item(), loss_meta_val.item()))

    writer.add_scalar('meta-train loss', loss_meta_train.item(), iterate)
    writer.add_scalar('meta-val loss', loss_meta_val.item(), iterate)
    writer.add_scalar('sound_mean val mse Loss', sound_mean_val_mse.item(), iterate)

    return iterate


def eval(criterion_meta_train, test_loader, image_sound_extractor,sound_mean, device, writer, encoder=None, mode=7):
    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    image_sound_extractor.eval()
    loss = 0
    total = 0
    for i, batch in enumerate(test_loader):

        audio = batch[0].to(device)
        meta = batch[1].to(device)
        video = batch[2].to(device)
        user_embeddings = batch[3].to(device)
        labels = batch[4].to(device)
        labels = torch.reshape(labels, (labels.shape[0], 1))

        outputs,_,_,_ = image_sound_extractor(labels, user_embeddings, audio, meta, video, encoder=encoder,  sound_mean=sound_mean, noise_layer=['fc0','fc1','fc2'], meta_train=False, mode=mode)

        loss += criterion_meta_train(outputs, labels) * labels.size(0)
        total += labels.size(0)
    rmse = torch.sqrt(loss/total)
    
    return rmse


if __name__ == '__main__':
    main(parse_args())

