import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import argparse
from tqdm import tqdm
import time
import os
from read_data import TinyImageNetDataSet
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cuda', help="")
parser.add_argument("--max_epoch", type=int, default=50, help="")
parser.add_argument("--data_dir", type=str, default='./tiny-imagenet-200/train', help="")
parser.add_argument("--im_list_file", type=str, default='./train_files_list.txt', help="")
parser.add_argument("--batch_size", type=int, default=1, help="")
parser.add_argument("--chckpnt_dir", type=str, default='./checkpoints', help="")
parser.add_argument("--tmp_dir", type=str, default='./temp', help="")
parser.add_argument("--log_file", type=str, default='./log.txt', help="")
opt = parser.parse_args()

if not os.path.exists(opt.chckpnt_dir): os.mkdir(opt.chckpnt_dir)
if not os.path.exists(opt.tmp_dir): os.mkdir(opt.tmp_dir)

train_dataset = TinyImageNetDataSet(data_dir=opt.data_dir, im_list_file=opt.im_list_file)
train_loader = DataLoader(dataset=train_dataset,  batch_size=opt.batch_size, shuffle=False, pin_memory=True)

for epoch in range(1, opt.max_epoch+1):

	ep_st = time.time()

	loader = tqdm(train_loader, total=len(train_loader))
	for itr, image in enumerate(loader):
		image = image.to(opt.device)            
		image_down = F.interpolate(image, scale_factor=0.5)

		print(image, '\n', image.size())
		print(image_down, '\n', image_down.size())

		save_image(image, f'{opt.tmp_dir}/image.png')
		save_image(image_down, f'{opt.tmp_dir}/image_down.png')

		exit()