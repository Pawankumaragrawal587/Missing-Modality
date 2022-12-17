import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

import pickle
import numpy as np
import pandas as pd
import os
import cv2 
import random
from PIL import Image
import math
import sys
sys.path.append("../")

from sklearn.model_selection import train_test_split
from os import path

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def generate_modality_priors(modality, index_list, name):
    print('Generating modality Prior for Modality:', modality)

    rating_list = pd.read_csv('../data/train.csv')
    movie_set = set()
    for i in index_list:
    	row = rating_list.iloc[[i], :]
    	movie_set.add(int(row['movie_id']-1))

    data = []
    if modality == 0:
        for i in movie_set:
            with open('../data/audioEmbed/' + str(i) + '.pkl', 'rb') as pickle_file:
                audio_embedding = pickle.load(pickle_file).to_numpy()
                audio_embedding = audio_embedding.astype(np.float32)
            data.append(audio_embedding.reshape(300*1024))
        data = np.array(data)
    elif modality == 1:
        data = pd.read_csv('../data/Meta/embeddings.csv')
        data = data.iloc[list(movie_set), :]
        data = data.to_numpy()
    elif modality == 2:
        for i in movie_set:
            with open('../data/videoEmbed/' + str(i) + '.pkl', 'rb') as pickle_file:
                video_embedding = pickle.load(pickle_file)
                video_embedding = video_embedding.astype(np.float32)
            data.append(video_embedding.reshape(30*1000))
        data = np.array(data)
    
    kmeans = KMeans(n_clusters=10, random_state=0).fit(data)
    mean = kmeans.cluster_centers_
    print(mean.shape)
    save_path = path.join("../Output/", name + '_' + str(modality) +'.npy')
    np.save(save_path, mean)
    
    print('Finished generating modality Prior for Modality:', modality)

def split_dataset(data, complete_ratio, complete_list):
    print('Splitting Dataset...')

    name =  str(complete_ratio) + '_' + str(complete_list[0]) + '_' + str(complete_list[1]) + '_' + str(complete_list[2])
    index = np.array(data.index)
    num = len(index)
    arr = np.array([7]*num)
    complete = np.random.choice(index, int(num * complete_ratio) // 100, replace=False)
    index = list(set(index)-set(complete))
    num = len(index)
    for i in range(3):
        temp = np.random.choice(index, int(num * (1-complete_list[i])), replace=False)
        for j in temp:
            arr[j] &= ~(1 << i)

        if os.path.exists('modality_prior_' + name + '_' + str(i) + '.npy'):
        	continue
        # Generate modality priors
        index_list = []
        for j in range(len(arr)):
        	if (arr[j] & (1<<i)):
        		index_list.append(j)
        generate_modality_priors(i, index_list, 'modality_prior_' + name)

    res = [[], [], [], [], [], [], [], []]
    for i in range(len(arr)):
        res[arr[i]].append(i)
        
    res = np.array([np.array(l) for l in res], dtype=object)
    
    save_path = path.join("../Output/", 'index_list_' + name + '.npy')
    np.save(save_path, res)

    print('Finished Splitting Dataset.')

class MetaTrSouMNIST(torch.utils.data.Dataset):
	"""  soundmnist dataset for meta-learning"""

	def __init__(self, mode, modality_complete_list, modality_complete_ratio):

		self.mode = mode

		self.user = pd.read_csv('../data/User/embeddings.csv')
		self.meta = pd.read_csv('../data/Meta/embeddings.csv')
		self.rating_list = pd.read_csv('../data/train.csv')

		index_file = '../Output/' + 'index_list_' + str(modality_complete_ratio) \
					   + '_' + str(modality_complete_list[0]) \
					   + '_' + str(modality_complete_list[1]) \
					   + '_' + str(modality_complete_list[2]) + '.npy'

		if not os.path.exists(index_file):
			split_dataset(self.rating_list, modality_complete_ratio, modality_complete_list)

		self.index_list = np.load(index_file, allow_pickle=True)[self.mode]

	def __len__(self):
		""" Returns size of the dataset
		returns:
			int - number of samples in the dataset
		"""
		return len(self.index_list)

	def __getitem__(self, index):
		""" get image and label  """
		transformations = transforms.Compose([transforms.ToTensor(),
											  transforms.Normalize([0.5], [0.5])])

		row = self.rating_list.iloc[[self.index_list[index]], :]

		rating = torch.tensor(int(row['rating'])/5, dtype=torch.float32)

		user_embedding = self.user.iloc[row['user_id']-1]
		user_embedding = user_embedding.astype(np.float32)
		user_embedding = torch.tensor(user_embedding.values)

		if (self.mode & (2 ** 0)):
			m_id = int(row['movie_id']-1)
			with open('../data/audioEmbed/' + str(m_id) + '.pkl', 'rb') as pickle_file:
				audio_embedding = pickle.load(pickle_file).to_numpy()
			audio_embedding = audio_embedding.astype(np.float32)
			audio_embedding = torch.tensor(audio_embedding)
		else:
			audio_embedding = torch.tensor(np.array([0]))

		if (self.mode & (2 ** 1)):
			meta_embedding = self.meta.iloc[row['movie_id']-1]
			meta_embedding = meta_embedding.astype(np.float32)
			meta_embedding = torch.tensor(meta_embedding.values)
		else:
			meta_embedding = torch.tensor(np.array([0]))

		if (self.mode & (2 ** 2)):
			with open('../data/videoEmbed/' + str(int(row['movie_id']-1)) + '.pkl', 'rb') as pickle_file:
				video_embedding = pickle.load(pickle_file)
			video_embedding = video_embedding.astype(np.float32)
			video_embedding = torch.tensor(video_embedding)
		else:
			video_embedding = torch.tensor(np.array([0]))

		return audio_embedding, meta_embedding, video_embedding, user_embedding, rating 
