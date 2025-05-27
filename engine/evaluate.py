import seaborn as sns
from sklearn.metrics import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cal_perf(true, pred, pos_proba, eps=1e-10):  # only binary
    acc, pre, rec, f1 = accuracy_score(true, pred), precision_score(true, pred), recall_score(true, pred), f1_score(true, pred)
    tn, fp, fn, tp = confusion_matrix(true, pred).flatten()
    spec = (tn + eps) / ((fp + tn) + eps)
    ppv = (tp + eps) / ((tp + fp) + eps)
    npv = (tn + eps) / ((tn + fn) + eps)
    #auc = roc_auc_score(true, pos_proba)
    try: 
        auc = roc_auc_score(true, pos_proba)
    except ValueError:
        non_except = ~np.isnan(pos_proba)
        true_filtered = np.array(true)[non_except]
        pos_proba_filtered = np.array(pos_proba)[non_except]
        auc = roc_auc_score(true_filtered, pos_proba_filtered)

    perf = [acc, pre, rec, spec, f1, auc, ppv, npv]
    return perf


def hard_vote(df, pred_col):
    hv_count_neg = len(df[df[pred_col] == 0])
    hv_count_pos = len(df[df[pred_col] == 1])
    hv_result = (hv_count_neg, hv_count_pos)
    #hv_pred = np.argmax(hv_result)
    return hv_result

def soft_vote(df, neg_prob_col, pos_prob_col):
    sv_prob_neg = np.mean(df[neg_prob_col])  # soft voting results (class0)
    sv_prob_pos = np.mean(df[pos_prob_col])  # soft voting results (class1)
    sv_prob = (sv_prob_neg, sv_prob_pos)
    #sv_pred = np.argmax(sv_prob)
    return sv_prob


# positive case
def get_pos_case(df, patient_list, thr=0.5):
    TARGET_CLASS = 1
    pos_vote_perf_patient = {'ID':[], 'Hard':[], 'Soft':[], 'H_pred':[], 'S_pred':[]}

    pos_df = df[df['class_id'] == TARGET_CLASS]
    pos_df = pos_df.sort_values(by='patient_id').reset_index(drop=True)

    for patient_idx in patient_list:
        one_case = pos_df[pos_df['patient_id'] == patient_idx]
        hv_result = hard_vote(one_case, 'pred')
        sv_prob = soft_vote(one_case, 'neg_prob', 'pos_prob')

        pos_vote_perf_patient['ID'].append(patient_idx)
        pos_vote_perf_patient['Hard'].append(hv_result)
        pos_vote_perf_patient['Soft'].append(sv_prob)
        pos_vote_perf_patient['H_pred'].append(np.argmax(hv_result))
        soft_pred = 1 if sv_prob[1] >= thr else 0
        pos_vote_perf_patient['S_pred'].append(soft_pred)

    pos_voting_df = pd.DataFrame(pos_vote_perf_patient)
    pos_voting_df['true'] = [TARGET_CLASS for _ in range(len(patient_list))]

    return pos_voting_df


# negative case
def get_neg_case(df, patient_list, thr=0.5):
    TARGET_CLASS = 0
    neg_vote_perf_patient = {'ID':[], 'Hard':[], 'Soft':[], 'H_pred':[], 'S_pred':[]}

    neg_df = df[df['class_id'] == TARGET_CLASS]
    neg_df = neg_df.sort_values(by='patient_id').reset_index(drop=True)

    for patient_idx in patient_list:
        one_case = neg_df[neg_df['patient_id'] == patient_idx]
        hv_result = hard_vote(one_case, 'pred')
        sv_prob = soft_vote(one_case, 'neg_prob', 'pos_prob')

        neg_vote_perf_patient['ID'].append(patient_idx)
        neg_vote_perf_patient['Hard'].append(hv_result)
        neg_vote_perf_patient['Soft'].append(sv_prob)
        neg_vote_perf_patient['H_pred'].append(np.argmax(hv_result))
        soft_pred = 1 if sv_prob[1] >= thr else 0
        neg_vote_perf_patient['S_pred'].append(soft_pred)

    neg_voting_df = pd.DataFrame(neg_vote_perf_patient)
    neg_voting_df['true'] = [TARGET_CLASS for _ in range(len(patient_list))]

    return neg_voting_df


def get_cm(y_true, y_pred, save_path, label_name: list):   # labels = ['LM', 'GM', 'SSc']
    num_cls = len(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5.5, 5))

    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percent = ["{0:.2%}".format(value) for value in (cm.flatten() / np.sum(cm))]
    labels = [f"{v1}\n\n({v2})" for v1, v2 in zip(group_counts, group_percent)]
    labels = np.asarray(labels).reshape(num_cls, num_cls)

    # cm/np.sum(cm))*70 여기서 곱하는 값을 바꿔서 색 조절
    f = sns.heatmap((cm / np.sum(cm)) * 70, annot=labels, fmt='',
                    cmap='Blues', vmin=0, vmax=25, linewidths=0.1,
                    annot_kws={'size': '12'}, cbar_kws={'label': '(%)'})  # annot=True, fmt='.2%'

    fig = f.figure
    cbar = fig.get_children()[-1]
    cbar.yaxis.set_ticks([0, 25])

    labels = label_name
    f.set_xticklabels(labels, fontdict={'size': '12'})
    f.set_yticklabels(labels, fontdict={'size': '12'})

    f.set(xlabel='Predicted label', ylabel='True label')
    f.axhline(y=0, color='k', linewidth=1)
    f.axhline(y=num_cls, color='k', linewidth=2)
    f.axvline(x=0, color='k', linewidth=1)
    f.axvline(x=num_cls, color='k', linewidth=2)

    plt.title("Confusion Matrix", fontsize=18, y=1.02)
    plt.xlabel('Predicted label', fontsize=12, labelpad=15)
    plt.ylabel('True label', fontsize=12, labelpad=14)

    plt.tight_layout()
    plt.savefig(save_path)