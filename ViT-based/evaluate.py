import os
import random
import shutil
import pickle
import argparse

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from dataset.augmentations import *
from dataset.load_data_for_CV import CustomDataset, get_IDs
from engine.loop import *
from engine.evaluate import *
from model import models_vit

from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


def main(args):
	# GPU setup
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f"# Using device: {device}")

	def set_random_seed(seed):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		
	# Set random seed
	set_random_seed(args.random_seed)
	
	#test_transform = non_aug(args.input_size)  # Validation typically uses non-augmented data
	
	# Load dataset 
	if args.infer_type == 'internal':
		# metadata = pd.read_csv(os.path.join(args.data_path, 'metadata.csv'))
		tr = pd.read_csv(os.path.join(args.data_path, 'train_annotation_file.csv'), dtype={'patient_id': str})
		va = pd.read_csv(os.path.join(args.data_path, 'valid_annotation_file.csv'), dtype={'patient_id': str})
		te = pd.read_csv(os.path.join(args.data_path, 'test_annotation_file.csv'), dtype={'patient_id': str})

		totalset = pd.concat([tr, va, te], axis=0).reset_index(drop=True)
		totalset['image_id'] = totalset['image_id'] + '.png'

	elif args.infer_type == 'external':
		extraset = pd.read_csv(os.path.join(args.data_path, 'extra_annotation_file.csv'), dtype={'patient_id': str})
		ex_patients = np.unique(list(extraset['patient_id']))[70:]
		
		extraset = extraset[extraset['patient_id'].isin(ex_patients)].reset_index(drop=True)
		extraset['image_id'] = extraset['image_id'] + '.png'
		
		# IRB 기준 미충족 케이스 제외
		extraset = extraset[extraset['patient_id'] != '202'].reset_index(drop=True)
		
		ex_p_ids, ex_inames, ex_labels = get_IDs(extraset, ex_patients)
		
		extra_dataset = CustomDataset(
			root_dir=os.path.join(args.data_path, 'extraset'),
			p_ids=ex_p_ids, 
			inames=ex_inames, 
			labels=ex_labels, 
			#transform=test_transform, 
			custom_aug=False
		)
		
		extra_loader = DataLoader(extra_dataset, 
								  batch_size=args.batch_size, 
								  num_workers=4, 
								  shuffle=False)
	
	
	save_path = f"{args.save_dir}/{args.augmentation}/proportion_{int(args.data_proportion*100)}"
	with open(os.path.join(save_path, f'cross_val_hist.pkl'), 'rb') as file:
		CV_hist = pickle.load(file)
	
	test_results = {}
	for fold in range(args.fold_num):
		if args.infer_type == 'internal':
			te_p_ids = CV_hist[str(fold)]['test']
			te_p_ids, te_inames, te_labels = get_IDs(totalset, np.unique(te_p_ids))

			test_dataset = CustomDataset(
				root_dir=os.path.join(args.data_path, 'dataset_all'),
				p_ids=te_p_ids, 
				inames=te_inames, 
				labels=te_labels, 
				#transform=test_transform, 
				custom_aug=False
			)

			test_loader = DataLoader(test_dataset, 
									 batch_size=args.batch_size, 
									 shuffle=False,
									 num_workers=4)			
		
		# Model setup
		model = models_vit.__dict__['vit_base_patch16'](
			num_classes=args.num_finetune_classes,
			drop_path_rate=0.1,
			global_pool=False,
		)
		#model.fc = nn.Identity() if args.TSNE else nn.Linear(model.fc.in_features, args.num_finetune_classes)
		
		checkpoint_path = os.path.join(save_path, f"Best_Fold{fold+1}.pth")
		if not os.path.exists(checkpoint_path):
			raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
			
		checkpoint = torch.load(checkpoint_path, map_location='cpu')
		model.load_state_dict(checkpoint, strict=False)
		model.to(device)
		
		# Loss
		criterion = nn.CrossEntropyLoss()
		
		# inference
		fold_results = {}
		
		if args.TSNE:
			features, labels = extract_features(test_loader if args.infer_type == 'internal' else extra_loader, model, device)
			fold_results['Features'] = features
			fold_results['Labels'] = labels
		else:
			_, prob, true = test_loop(test_loader if args.infer_type == 'internal' else extra_loader, model, criterion, device)
			pred = np.argmax(prob, axis=1)
			fold_results['True'] = true
			fold_results['Pred'] = pred
			fold_results['Prob'] = prob		
		
# 		if args.infer_type == 'internal':
# 			_ , prob, true = test_loop(test_loader, model , criterion, device)   
# 		elif args.infer_type == 'external':
# 			_ , prob, true = test_loop(extra_loader, model , criterion, device) 
			
# 		pred = np.argmax(prob, axis=1)
# 		fold_results['True'] = true
# 		fold_results['Pred'] = pred
# 		fold_results['Prob'] = prob
		
		test_results[fold] = fold_results
		torch.cuda.empty_cache()
	
	if args.TSNE:
		if args.infer_type == 'internal':
			with open(os.path.join(save_path, f'test_tsne_results.pkl'), 'wb') as file:
				pickle.dump(test_results, file)
		elif args.infer_type == 'external':
			with open(os.path.join(save_path, f'extra_tsne_results.pkl'), 'wb') as file:
				pickle.dump(test_results, file)

	else:
		if args.infer_type == 'internal':
			with open(os.path.join(save_path, f'test_results.pkl'), 'wb') as file:
				pickle.dump(test_results, file)
		elif args.infer_type == 'external':
			with open(os.path.join(save_path, f'extra_results.pkl'), 'wb') as file:
				pickle.dump(test_results, file)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Training Script with 5-Fold Cross Validation")
	parser.add_argument('--gpu', type=str, default='0', help='GPU number to use')
	parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
	parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
	parser.add_argument('--fold_num', type=int, default=5)
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
	parser.add_argument('--num_finetune_classes', type=int, default=2, help='Number of fine-tuning classes')
	parser.add_argument('--input_size', type=int, default=224, help='Input image size')
	parser.add_argument('--data_proportion', type=float, default=1.0, help='Proportion of data to use for training')
	parser.add_argument('--augmentation', type=str, required=True, help='Augmentation strategy to use')
	parser.add_argument('--save_dir', type=str, required=True, help='Directory to save models and logs')
	parser.add_argument('--infer_type', type=str, required=True, choices=["internal", "external"])
	parser.add_argument('--TSNE', type=bool, default=False)
	
	args = parser.parse_args()
	main(args)
