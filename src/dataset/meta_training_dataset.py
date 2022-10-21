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
from utils.wav2mfcc import wav2mfcc 

from sklearn.model_selection import train_test_split

def split_and_save(modality_complete_ratio):
	training_data = pd.read_csv('../data/train.csv')
	if modality_complete_ratio == 100:
		meta_train = pd.DataFrame()
		meta_val = training_data
	elif modality_complete_ratio == 0:
		meta_val = pd.DataFrame()
		meta_train = training_data
	else:
		meta_train, meta_val = train_test_split(training_data, test_size=modality_complete_ratio/100)
	meta_train.to_csv('../data/'+ 'meta_train_' + str(modality_complete_ratio) + '.csv')
	meta_val.to_csv('../data/'+ 'meta_val_' + str(modality_complete_ratio) + '.csv')

class MetaTrSouMNIST(torch.utils.data.Dataset):
	"""  soundmnist dataset for meta-learning"""

	def __init__(self, meta_split='mtr', modality_complete_ratio=100):

		self.meta_split = meta_split
		self.user = pd.read_csv('../data/User/embeddings.csv')
		self.meta = pd.read_csv('../data/Meta/embeddings.csv')

		if self.meta_split == 'mtr':
			rating_file_path = '../data/'+ 'meta_train_' + str(modality_complete_ratio) + '.csv'
			if not os.path.exists(rating_file_path):
				split_and_save(modality_complete_ratio)
			self.rating_list = pd.read_csv(rating_file_path)
			
		elif self.meta_split == 'mval':
			rating_file_path = '../data/'+ 'meta_val_' + str(modality_complete_ratio) + '.csv'
			if not os.path.exists(rating_file_path):
				split_and_save(modality_complete_ratio)
			self.rating_list = pd.read_csv(rating_file_path)

		else:
			raise ValueError('No such split: %s' % self.meta_split)

	def __len__(self):
		""" Returns size of the dataset
		returns:
			int - number of samples in the dataset
		"""
		return len(self.rating_list)

	def __getitem__(self, index):
		""" get image and label  """
		transformations = transforms.Compose([transforms.ToTensor(),
											  transforms.Normalize([0.5], [0.5])])

		row = self.rating_list.iloc[[index], :]

		rating = torch.tensor(int(row['rating'])/5, dtype=torch.float32)

		user_embedding = self.user.iloc[row['user_id']-1]
		user_embedding = user_embedding.astype(np.float32)
		user_embedding = torch.tensor(user_embedding.values)
		# user_embedding = transformations(user_embedding)

		with open('../data/videoEmbed/' + str(int(row['movie_id']-1)) + '.pkl', 'rb') as pickle_file:
			video_embedding = pickle.load(pickle_file)
		video_embedding = video_embedding.astype(np.float32)
		video_embedding = torch.tensor(video_embedding)
		# video_embedding = transformations(video_embedding)

		if self.meta_split == 'mtr':
			return video_embedding, user_embedding, rating 
			# return video_embedding, rating 

		elif self.meta_split == 'mval':
			meta_embedding = self.meta.iloc[row['movie_id']-1]
			meta_embedding = meta_embedding.astype(np.float32)
			meta_embedding = torch.tensor(meta_embedding.values)
			# meta_embedding = transformations(meta_embedding)

			return video_embedding, meta_embedding, user_embedding, rating 
			# return video_embedding, meta_embedding, rating 

		else:
			raise ValueError('No such split: %s' % self.meta_split)


if __name__ == '__main__':
	from PIL import Image
	import torch
	img_root = '../data/mnist/'
	sound_root = '../data/sound_450/'

	dataset = MetaTrSouMNIST(img_root, sound_root, per_class_num=105, meta_split='mtr')
	sampler = torch.utils.data.RandomSampler(dataset, replacement=True)
	loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=1, pin_memory=True,sampler=sampler)
	
	print(len(dataset))

