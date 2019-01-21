import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
import torchvision.transforms as transforms
from torchvision.utils import save_image
import argparse
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
import numpy as np

from read_data import TinyImageNetDataSet
from networks import Generator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cuda', help="")
parser.add_argument("--max_epoch", type=int, default=100, help="")
parser.add_argument("--data_dir", type=str, default='../../../data/tiny-imagenet-200/train', help="")
parser.add_argument("--im_list_file", type=str, default='../../../data/tiny-imagenet-200/train_files_list.txt', help="")
parser.add_argument("--batch_size", type=int, default=16, help="")
parser.add_argument("--chckpnt_dir", type=str, default='./checkpoints', help="")
parser.add_argument("--tmp_dir", type=str, default='./temp', help="")

parser.add_argument("--org_im_size", type=int, default=64, help='number of pixels in the original image')
parser.add_argument("--scale_factor", type=float, default=0.5, help='scaling factor for low resolution images')
parser.add_argument("--n_ch", type=int, default=3, help='number of channels of the images')
parser.add_argument("--first_dim", type=int, default=4, help='first dimension of the image before upsampling operations (after vectorization and reshape)')
parser.add_argument("--ngf", type=int, default=512, help='number of filters for the first convolution of the generator')
parser.add_argument("--ndf", type=int, default=128, help='number of filters for the first convolution of the discriminator')
parser.add_argument("--n_layers", type=int, default=3, help='number of filters for the first convolution of the discriminator')

parser.add_argument("--lambda_im", type=float, default=50.0, help='coefficient of the image loss for the calculation of total Generator loss')
parser.add_argument("--lr", type=float, default=2e-4, help='learning rate')


opt = parser.parse_args()

if not os.path.exists(opt.chckpnt_dir): os.mkdir(opt.chckpnt_dir)
if not os.path.exists(opt.tmp_dir): os.mkdir(opt.tmp_dir)

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_dataset = TinyImageNetDataSet(data_dir=opt.data_dir, 
									im_list_file=opt.im_list_file, 
									out_size=int(opt.org_im_size * opt.scale_factor),
									transform = transforms.Compose([transforms.ToTensor(), 
																	normalize # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L188
																	]))
train_loader = DataLoader(dataset=train_dataset,  batch_size=opt.batch_size, shuffle=False, pin_memory=True)

net_G = Generator(opt=opt).to(opt.device)
net_D = Discriminator(opt=opt).to(opt.device)

optim_G = torch.optim.Adam(net_G.parameters(), lr=opt.lr)
optim_D = torch.optim.SGD(net_D.parameters(), lr=opt.lr)

net_G.train()
net_D.train()

print(net_G)
print('\n\n',net_D)

adv_loss = nn.MSELoss().to(opt.device)
im_loss = nn.L1Loss().to(opt.device)

for epoch in range(1, opt.max_epoch+1):

	ep_st = time.time()

	loader = tqdm(train_loader, total=len(train_loader))
	d_l, g_l, im_l, counter = 0, 0, 0, 0
	for itr, (image, image_down) in enumerate(loader):
		image = image.to(opt.device)            
		image_down = image_down.to(opt.device)

		fake = net_G(image_down)
		
		net_D.zero_grad()
		D_fake = net_D(fake.detach())
		D_real = net_D(image)

		D_Loss = 0.5 * (adv_loss(D_real, torch.ones_like(D_real)) + adv_loss(D_fake, torch.zeros_like(D_fake)))
		D_Loss.backward()
		optim_D.step() 

		net_G.zero_grad()
		GAN_G_loss = adv_loss(net_D(fake), torch.ones_like(D_fake))
		Im_Loss = im_loss(fake, image)
		G_Loss = GAN_G_loss + opt.lambda_im * Im_Loss
		G_Loss.backward()
		optim_G.step()

		g_l += GAN_G_loss.item()
		d_l += D_Loss.item()
		im_l += Im_Loss.item()
		counter += 1

	ep_end = time.time()

	g_l /= counter
	d_l /= counter
	im_l /= counter

	G_LOSSES.append(g_l)
	D_LOSSES.append(d_l)
	IM_LOSSES.append(im_l)

	print(f'epoch: {epoch:03d}/{opt.max_epoch}:\n'\
		 +f'Gen Loss: {g_l:.4f} | Dis Loss: {d_l:.4f} | Image Loss: {im_l:.4f}'\
		 +f'\tin {int(ep_end-ep_st)} sec')

	### Save example images
	nrow = int(round(np.sqrt(opt.batch_size)))
	save_image(image*0.5+0.5, f'{opt.tmp_dir}/{epoch}_image.png', nrow=nrow)
	save_image(image_down*0.5+0.5, f'{opt.tmp_dir}/{epoch}_image_down.png', nrow=nrow)
	save_image(fake*0.5+0.5, f'{opt.tmp_dir}/{epoch}_fake.png', nrow=nrow)

