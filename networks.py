"""
Generator and Discriminator architectures are inspired from pix2pix network:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
"""
import torch
import torch.nn as nn 
import math

class Generator(nn.Module):
	def __init__(self, opt):
		super(Generator, self).__init__()

		self.opt = opt
		dim = int(opt.org_im_size * opt.scale_factor)		
		self.linear = nn.Linear(dim*dim*opt.n_ch, opt.ngf*opt.first_dim*opt.first_dim)
		
		self.first_conv = nn.Conv2d(opt.ngf, opt.ngf, kernel_size=3, stride=1, padding=1)		
		
		n_ups = int(math.log(opt.org_im_size, 2)) - int(math.log(opt.first_dim, 2))
		self.up_layers = nn.ModuleList()
		ch_in = opt.ngf
		ch_out = opt.ngf // 2
		for i in range(n_ups):
			self.up_layers.append(nn.ReLU(True))
			self.up_layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1))
			self.up_layers.append(nn.BatchNorm2d(ch_out))

			ch_in = ch_out
			ch_out = ch_out //2

		self.last_relu = nn.ReLU(True)
		self.last_conv = nn.Conv2d(opt.ngf>>n_ups, opt.n_ch, kernel_size=3, stride=1, padding=1)
		self.tanh = nn.Tanh()

	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.linear(x)
		x = x.view(x.size(0), self.opt.ngf, self.opt.first_dim, self.opt.first_dim)
		x = self.first_conv(x)
		for layer in self.up_layers:
			x = layer(x)

		x = self.last_relu(x)
		x = self.last_conv(x)
		return self.tanh(x)


class Discriminator(nn.Module):
	def __init__(self, opt):
		super(Discriminator, self).__init__()

		self.opt = opt
		self.layers = nn.ModuleList()
		ch_in = opt.n_ch
		ch_out = opt.ndf				
		for i in range(opt.n_layers):
			self.layers.append(nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1))
			self.layers.append(nn.LeakyReLU(0.2, True))
			self.layers.append(nn.Dropout2d(0.25))
			self.layers.append(nn.BatchNorm2d(ch_out))
			ch_in = ch_out
			ch_out = ch_in*2

		ch_out = ch_out // 2
		self.last = nn.Linear(ch_out*(opt.org_im_size>>opt.n_layers)*(opt.org_im_size>>opt.n_layers), 1)

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		x = self.last(x.view(x.size(0), -1))
		return x








