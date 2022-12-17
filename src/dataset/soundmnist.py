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
# import scipy.io as scio
from PIL import Image
import math
import sys
sys.path.append("../")


class SoundMNIST(torch.utils.data.Dataset):
	"""  soundmnist dataset """

	def __init__(self, img_root, sound_root, per_class_num=105,train=True):

		self.user = pd.read_csv('../data/User/embeddings.csv')
		self.meta = pd.read_csv('../data/Meta/embeddings.csv')
		self.rating_list = pd.read_csv('../data/test.csv')

	def __len__(self):
		""" Returns size of the dataset
		returns:
			int - number of samples in the dataset
		"""
		return len(self.rating_list)

	def __getitem__(self, index):
		""" get image and label  """
		# transformations = transforms.Compose([transforms.ToTensor(),
		# 									  transforms.Normalize([0.5], [0.5])])

		row = self.rating_list.iloc[[index], :]

		rating = torch.tensor(int(row['rating'])/5, dtype=torch.float32)

		user_embedding = self.user.iloc[row['user_id']-1]
		user_embedding = user_embedding.astype(np.float32)
		user_embedding = torch.tensor(user_embedding.values)
		# user_embedding = transformations(user_embedding)

		with open('../data/audioEmbed/' + str(int(row['movie_id']-1)) + '.pkl', 'rb') as pickle_file:
			audio_embedding = pickle.load(pickle_file).to_numpy()
		audio_embedding = audio_embedding.astype(np.float32)
		audio_embedding = torch.tensor(audio_embedding)
		# video_embedding = transformations(video_embedding)

		meta_embedding = self.meta.iloc[row['movie_id']-1]
		meta_embedding = meta_embedding.astype(np.float32)
		meta_embedding = torch.tensor(meta_embedding.values)

		with open('../data/videoEmbed/' + str(int(row['movie_id']-1)) + '.pkl', 'rb') as pickle_file:
			video_embedding = pickle.load(pickle_file)
		video_embedding = video_embedding.astype(np.float32)
		video_embedding = torch.tensor(video_embedding)

		return audio_embedding, meta_embedding, video_embedding, user_embedding, rating
		# return video_embedding, meta_embedding, rating

