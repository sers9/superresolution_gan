from torch.utils.data import Dataset
from PIL import Image

def check_gray_scale(image, path):
	if image.mode == 'RGB':
		return image
	elif image.mode == 'L':
		# print('\ngray scale image: ', path)
		rgbimg = Image.new("RGB", image.size)
		rgbimg.paste(image)
		return rgbimg

class TinyImageNetDataSet(Dataset):
	def __init__(self, data_dir, im_list_file, out_size, transform=None):
		"""
		Args:
			data_dir (string): Path to the image files.
			im_list_file (string): txt file that lists all of the image files under data_dir with paths including subfolder names
			transform (callable, optional): Optional transform to be applied on a sample.
		"""
		self.data_dir = data_dir
		with open(im_list_file) as f:
			self.im_list = f.read().splitlines()
		self.out_size = out_size
		self.transform = transform

	def __getitem__(self, index):

		img_path = self.im_list[index]
		img = Image.open(img_path)
		img = check_gray_scale(img, img_path)
		img_down = img.resize(size=(self.out_size, self.out_size))

		if self.transform:
			img = self.transform(img)
			img_down = self.transform(img_down)
		return img, img_down
 
	def __len__(self):
		return len(self.im_list)