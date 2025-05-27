import os
import random
import shutil
import pickle

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from engine.evaluate import *
from sklearn.metrics import *


# Hard voting
def get_hard_perf(voting_df):
	pos_proba = [v[1] for v in voting_df['Hard']]
	hard_perf = cal_perf(voting_df['true'], voting_df['H_pred'], pos_proba)
	return hard_perf

# Soft voting
def get_soft_perf(voting_df):
	pos_proba = [v[1] for v in voting_df['Soft']]
	soft_perf = cal_perf(voting_df['true'], voting_df['S_pred'], pos_proba)
	return soft_perf


def get_image_perf(fold, results, save_path):
	fold_results = results[fold]
	
	true, pred, pos_proba = fold_results['True'], fold_results['Pred'], fold_results['Prob'][:,1]
	perf = cal_perf(true, pred, pos_proba)
	
	get_cm(true, pred, os.path.join(save_path, f'fold{fold}_cm.png'), ['Negative', 'Positive'])
	return perf


def get_region_perf(fold, patient_list, results):
	unique_patient_list = np.unique(patient_list)
	
	df = pd.DataFrame({
		'patient_id': patient_list,  
		'class_id': results[fold]['True'],
		'pred': results[fold]['Pred'],
		'neg_prob': results[fold]['Prob'][:, 0],
		'pos_prob': results[fold]['Prob'][:, 1]
	})
	
	pos_voting_df = get_pos_case(df, unique_patient_list)
	neg_voting_df = get_neg_case(df, unique_patient_list)
	voting_df = pd.concat([pos_voting_df, neg_voting_df], axis=0).reset_index(drop=True)
	print("Num of test cases (pos + neg) : ", len(voting_df))
	
	return voting_df
