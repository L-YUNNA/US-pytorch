import os
import json
import pickle
import gc
import random
import argparse
import pandas as pd

import torch
from torch import nn
from torchvision import utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from dataset.augmentations import *
from dataset.load_data_for_CV import *
from engine.loop import *
from model import models_vit

from sklearn.metrics import *
from sklearn.model_selection import KFold, train_test_split


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

	
	# Load dataset 
	# metadata = pd.read_csv(os.path.join(args.data_path, 'metadata.csv'))
	tr = pd.read_csv(os.path.join(args.data_path, 'train_annotation_file.csv'), dtype={'patient_id': str})
	va = pd.read_csv(os.path.join(args.data_path, 'valid_annotation_file.csv'), dtype={'patient_id': str})
	te = pd.read_csv(os.path.join(args.data_path, 'test_annotation_file.csv'), dtype={'patient_id': str})

	totalset = pd.concat([tr, va, te], axis=0).reset_index(drop=True)
	totalset['image_id'] = totalset['image_id'] + '.png'
	patients = np.unique(totalset['patient_id'].tolist())
	
	
	# Dynamically select augmentation strategy
	augmentations = {
		"Non": non_aug,
		"Baseline": base_aug,
		"VSPA": custom_aug,
		"Combined": combined_aug,
	}
	train_transform = augmentations[args.augmentation](args.input_size)
	test_transform = non_aug(args.input_size)  # Validation typically uses non-augmented data
	
	if args.augmentation == 'VSPA' or 'Combined':
		run_custom_aug = True

	
	# 5-Fold Cross Validation
	kf_outer = KFold(n_splits=args.fold_num, shuffle=True, random_state=args.random_seed)
	
	CV_hist = {}
	for fold, (train_index, test_index) in enumerate(kf_outer.split(patients, patients)):
		print(f"# Starting Fold {fold+1}...")
		fold_hist = {'train':[], 'valid':[], 'test':[]}
		
		tr_patients, te_patients = patients[train_index], patients[test_index]
		tr_patients, va_patients, _, _ = train_test_split(tr_patients, tr_patients, 
														  test_size=0.25, 
														  random_state=args.random_seed)
		
		print(f"[Org] train : {len(tr_patients)}, valid : {len(va_patients)}, test : {len(te_patients)}")
		tr_patients = np.random.choice(tr_patients, size=int(args.data_proportion*len(tr_patients)), replace=False)
		va_patients = np.random.choice(va_patients, size=int(args.data_proportion*len(va_patients)), replace=False)
		print(f"[Sampled] train : {len(tr_patients)}, valid : {len(va_patients)}, test : {len(te_patients)}")
		
		
		save_path = f"{args.save_dir}/{args.augmentation}/proportion_{int(args.data_proportion*100)}"
		os.makedirs(save_path, exist_ok=True)
		
		with open(os.path.join(save_path, f"Fold{fold+1}_log.txt"), 'w') as log_file:
			# get train, valid, test info
			tr_p_ids, tr_inames, tr_labels = get_IDs(totalset, tr_patients)
			va_p_ids, va_inames, va_labels = get_IDs(totalset, va_patients)
			te_p_ids, te_inames, te_labels = get_IDs(totalset, te_patients)
			fold_hist['train'].append(tr_p_ids)
			fold_hist['valid'].append(va_p_ids)
			fold_hist['test'].append(te_p_ids)

			train_dataset = CustomDataset(
				root_dir=os.path.join(args.data_path, 'dataset_all'),
				p_ids=tr_p_ids,
				inames=tr_inames,
				labels=tr_labels,
				transform=train_transform,
				custom_aug=run_custom_aug
			)

			val_dataset = CustomDataset(
				root_dir=os.path.join(args.data_path, 'dataset_all'),
				p_ids=va_p_ids,
				inames=va_inames,
				labels=va_labels,
				transform=test_transform,
				custom_aug=False
			)

			train_loader = DataLoader(train_dataset, 
									  batch_size=args.batch_size, 
									  shuffle=True, 
									  num_workers=4)
				  
			val_loader = DataLoader(val_dataset, 
									batch_size=args.batch_size, 
									shuffle=False, 
									num_workers=4)

			# Model setup
			model = timm.models.resnet50()
				  
			if args.pre_trained:
				if os.path.exists(args.pre_trained):
					pre_trained = torch.load(args.pre_trained, map_location='cpu')
					model.load_state_dict(pre_trained, strict=False)
					print(f"Loaded pre-trained weights from {args.pre_trained}")
				else:
					raise FileNotFoundError(
						f"The specified pre-trained weights file does not exist: {args.pre_trained}"
					)
			else:
				print("No pre-trained weights provided.")
				  
			model.fc = nn.Linear(in_features=model.fc.in_features, out_features=args.num_finetune_classes)
			model.to(device)

			# Loss and optimizer
			criterion = nn.CrossEntropyLoss()
			optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

			# Training loop for this fold
			best_score = 1
			max_epoch = 0
			for epoch in range(args.epochs): 
				current_lr = optimizer.param_groups[0]['lr']
				
				train_loss = train_loop(train_loader, model, criterion, optimizer, device)
				valid_loss, prob, true = test_loop(val_loader, model, criterion, device)
				  
				pred = np.argmax(prob, axis=1) 
				acc = accuracy_score(true, pred)

				print(f"Epoch ({epoch}/{args.epochs}) | Train loss: {train_loss:.4f}, \
					    Valid loss: {valid_loss:.4f}, Valid accuracy: {acc:.4f}")

				# Save model checkpoint
				if best_score > valid_loss:
					patience = 0
					best_score = valid_loss
					best_acc = acc
					max_epoch = epoch
					torch.save(model.state_dict(), os.path.join(save_path, f"Best_Fold{fold+1}.pth"))
					print("# BEST EPOCH : ", epoch)
					print(f"# BEST scores (loss, acc): {best_score, best_acc}")
				  
				# 로그 기록 추가
				log_entry = {
					"train_lr": current_lr,
					"train_loss": train_loss,
					"test_loss": valid_loss,
					"test_acc": acc * 100,
					"epoch": epoch,
					"max_epoch": max_epoch
				}
				log_file.write(json.dumps(log_entry) + '\n')  # 각 로그 엔트리를 JSON 형식으로 파일에 저장

			CV_hist[f'{fold}'] = fold_hist

			gc.collect()
			torch.cuda.empty_cache()
			print(f"# Completed Fold {fold+1}.")

	with open(os.path.join(save_path, f'cross_val_hist.pkl'), 'wb') as file:
		pickle.dump(CV_hist, file)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Training Script with 5-Fold Cross Validation")
	parser.add_argument('--gpu', type=str, default='0', help='GPU number to use')
	parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
	parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
	parser.add_argument('--fold_num', type=int, default=5)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
	parser.add_argument('--num_finetune_classes', type=int, default=2, help='Number of fine-tuning classes')
	parser.add_argument('--input_size', type=int, default=224, help='Input image size')
	parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
	parser.add_argument('--data_proportion', type=float, default=1.0, help='Proportion of data to use for training')
	parser.add_argument('--augmentation', type=str, required=True, 
						choices=["Non", "Baseline", "VSPA", "Combined"],
						help='Augmentation strategy to use')
	parser.add_argument('--pre_trained', type=str, default=False, help='Path to the pre-trained weight')
	parser.add_argument('--save_dir', type=str, required=True, help='Directory to save models and logs')
	
	args = parser.parse_args()
	main(args)
