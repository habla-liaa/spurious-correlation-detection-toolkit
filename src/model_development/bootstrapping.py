import numpy as np
from sklearn.utils import resample
from src.model_development.metrics import group_predictions, calculate_acc_score, calculate_auc_score
import tqdm
import joblib
import pandas as pd

from src.utils import save_pickle


def get_name(**kwargs):
    name = 'bootstrapped_metrics'
    name += f"-n_{kwargs['n_bootstraps']}"
    if kwargs.get('stratify'):
        name += '-stratified'
    return name


def get_repeated_indices(query, groups):
    group_to_idx = {group: idx for idx, group in enumerate(groups)}
    return np.array([group_to_idx[group] for group in query], dtype=int)


def calculate_bootstrapped_metrics(seed, labels, predicts, groups, stratify=False):
    if stratify:
        group_bootstrapped_sample = resample(groups, replace=True, n_samples=len(groups), random_state=seed, stratify=labels)                                               
    else:
        group_bootstrapped_sample = resample(groups, replace=True, n_samples=len(groups), random_state=seed)
    indexes = get_repeated_indices(group_bootstrapped_sample, groups)
    boot_labels = labels[indexes]
    boot_preds  = predicts[indexes]
    acc_value, random_acc = calculate_acc_score(boot_labels, boot_preds)
    auc_value = calculate_auc_score(boot_labels, boot_preds)
    return auc_value, acc_value, random_acc


def compute_bootstrapping(experiment_path, params):    
    bootstrapped_results = []
    seed_dirs = sorted(experiment_path.glob('seed_*'), key=lambda p: int(p.name.split('_')[-1]))
    for seed_dir in tqdm.tqdm(seed_dirs, desc=f"Bootstrapping was performed {params['n_bootstraps']} times for each seed"):
        seed_number = int(seed_dir.name.split('_')[-1])

        scores_filenames = sorted(seed_dir.glob('scores-test-f_*.pkl'))
        if not scores_filenames:
            continue

        labels_across_folds = []
        predicts_across_folds = []
        groups_across_folds = []
        
        for filename in scores_filenames:
            fold_labels, fold_predicts, fold_groups, _ = joblib.load(filename, mmap_mode='r')
            labels_across_folds.append(fold_labels)
            predicts_across_folds.append(fold_predicts)
            groups_across_folds.append(fold_groups)

        experiment_labels = np.concatenate(labels_across_folds, axis=0)
        experiment_predicts = np.concatenate(predicts_across_folds, axis=0)
        experiment_groups = np.concatenate(groups_across_folds, axis=0)

        grouped_labels, grouped_predicts, unique_groups = group_predictions(experiment_labels, experiment_predicts, experiment_groups)        
       
        for boot_id in range(params['n_bootstraps']):
            auc_value, acc_value, random_acc = calculate_bootstrapped_metrics(boot_id+seed_number, grouped_labels, grouped_predicts, 
                                                                              unique_groups, stratify=params.get('stratify', False))
            
            bootstrapped_results.append({
                'seed': seed_number,
                'bootstrapping_id': boot_id,
                'auc': auc_value,
                'acc': acc_value,
                'random_acc': random_acc,
            })
        
    return pd.DataFrame(bootstrapped_results)
    