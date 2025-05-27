import os
import random

from PIL import Image
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from dataset.custom_aug import PartialResizer


def load_image(image_path, transform=None):
	image = Image.open(image_path).convert("RGB")
	if transform:
		image = transform(image)
	image = image.unsqueeze(0)  # Add batch dimension
	return image


def get_IDs(df, patient_indices):
	p_ids = df[df['patient_id'].isin(patient_indices)]['patient_id'].tolist()
	inames = df[df['patient_id'].isin(patient_indices)]['image_id'].tolist()
	labels = df[df['patient_id'].isin(patient_indices)]['class_id'].tolist()
	return p_ids, inames, labels


def get_scales(ori_size, min_ratio, max_ratio):
	shrink = list(range(int(ori_size*min_ratio), ori_size+1, 10))
	expand = list(range(ori_size, int(ori_size*max_ratio), 25))
	image_scales = set()  # 중복을 허용하지 않는 set 사용

	for shrink_size, expand_size in zip(shrink, expand):
		# x축 축소
		image_scales.add((shrink_size, ori_size))
		# y축 축소
		image_scales.add((ori_size, shrink_size))
		# x축 확대
		image_scales.add((expand_size, ori_size))
		
		for sh_size in shrink:
			# x,y축 축소
			if sh_size != shrink_size:
				chosen_scale = random.choice([(shrink_size, sh_size), (sh_size, shrink_size)])
				image_scales.add(chosen_scale)
	return list(image_scales)


def transfrom_android(image, interpol_type=cv2.INTER_LINEAR):
	image = np.array(image)
	image = cv2.resize(image, (224, 224), interpolation=interpol_type)
	image = image.astype(np.float32)/255.0

	mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
	std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
	image = (image-mean)/std

	image = np.transpose(image, (2,0,1))
	#image = np.expand_dims(image, axis=0)

	image = torch.from_numpy(image)
	return image 


class CustomDataset(Dataset):
	def __init__(self, root_dir, p_ids, inames, labels, transform=None, input_size=(500, 500), custom_aug=False):
		self.root_dir = root_dir  # ../data/dataset_all
		self.p_ids = p_ids		# ['001', '002', ...]
		self.inames = inames	  # ['20221031_185216', '20221031_185257', ..]
		self.labels = labels	  # [1, 1, 0,...]
		self.transform = transform
		self.input_size = input_size
		self.custom_aug = custom_aug

		self.data = []
		for idx, class_name in enumerate(self.labels):
			img_path = os.path.join(root_dir, p_ids[idx], inames[idx])
			self.data.append((img_path, class_name))
		
		if self.custom_aug:
			self.various_image_scales = get_scales(ori_size=self.input_size[0], 
												   min_ratio=0.7, 
												   max_ratio=1.5)
			self.various_image_scales += [self.input_size] * len(self.various_image_scales)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path, label = self.data[idx]
		image = Image.open(img_path)
		if len(np.array(image).shape) != 3:
			image = image.convert('RGB')
		
		#if self.custom_aug:
		if self.custom_aug and random.random() < 0.7:
			partial_resize_augmentation = PartialResizer()
			
			size = random.choice(self.various_image_scales)
			#size = (224, 400)  # aug test용
		
			right_side_pad = self.input_size[0] - size[1]
			if right_side_pad < 0:
				right_side_pad = 0
			bottom_side_pad = self.input_size[0] - size[0]
			
			custom_transform = transforms.Compose([
				transforms.Resize(size),
				transforms.Pad((0, 0, right_side_pad, bottom_side_pad)),
				transforms.CenterCrop(self.input_size),
				
				transforms.RandomApply([
					partial_resize_augmentation,
				], p=0.7),
			])
			image = custom_transform(image)

		if isinstance(image, np.ndarray):
			image = Image.fromarray(image)
		
		if self.transform:
			image = self.transform(image)
			
		image = transfrom_android(image, interpol_type=cv2.INTER_LINEAR)

		return image, label