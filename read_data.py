from torch.utils.data import Dataset
from skimage.io import imread
import os
import numpy as np
import torchvision.transforms as transforms


class TinyImageNetDataSet(Dataset):
	def __init__(self, data_dir, im_list_file, transform=None):
		"""
		Args:
			data_dir (string): Path to the image files.
			im_list_file (string): txt file that lists all of the image files under data_dir with paths including subfolder names
			transform (callable, optional): Optional transform to be applied on a sample.
		"""
		self.data_dir = data_dir
		with open(im_list_file) as f:
			self.im_list = f.read().splitlines()

		self.transform = transform

	def __getitem__(self, index):

		img_path = self.im_list[index]
		img = imread(img_path) 
		img_ = transforms.ToTensor()(img)
		
		return img_
 
	def __len__(self):
		return len(self.im_list)