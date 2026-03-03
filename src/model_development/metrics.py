import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import roc_auc_score, accuracy_score
from src.utils import log


def calculate_auc_score(labels, predicts):
    if np.unique(labels).shape[0] < 2:
        return np.nan
    if predicts.shape[1] == 2:
        return roc_auc_score(labels, predicts[:, 1])
    else:
        return roc_auc_score(labels, predicts, multi_class='ovo')


def calculate_acc_score(labels, predicts):
    random_acc = np.unique(labels, return_counts=True)[1].max() / len(labels)
    return accuracy_score(labels, predicts.argmax(axis=1)), random_acc


def group_predictions(labels, predicts, groups):
    unique_groups, inverse_indices = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)
    n_classes = predicts.shape[1]

    grouped_predicts = np.zeros((n_groups, n_classes), dtype=np.float64)
    grouped_labels = np.zeros(n_groups, dtype=labels.dtype)

    for i in range(n_groups):
        mask = inverse_indices == i
        grouped_predicts[i] = predicts[mask].mean(axis=0)
        group_labels = np.unique(labels[mask])
        if group_labels.shape[0] != 1:
            raise ValueError(f"Group {unique_groups[i]} has inconsistent labels: {group_labels}")
        grouped_labels[i] = group_labels[0]

    return grouped_labels, grouped_predicts, unique_groups


def compute_metrics(experiment_path, indent):
    experiment_results = []
    
    log("COMPUTING METRICS", indent=indent)
    split = 'test'
    
    for seed_dir in sorted(experiment_path.glob('seed_*'), key=lambda p: int(p.name.split('_')[-1])):
        seed_number = int(seed_dir.name.split('_')[-1])

        scores_filenames = sorted(seed_dir.glob(f'scores-{split}-f_*.pkl'))
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

        grouped_labels, grouped_predicts, _ = group_predictions(
            experiment_labels,
            experiment_predicts,
            experiment_groups,
        )
        acc, random_acc = calculate_acc_score(grouped_labels, grouped_predicts)
            
        exp_res = {
            'seed': seed_number,
            'split': split,
            'auc': calculate_auc_score(grouped_labels, grouped_predicts),
            'acc': acc,
            'random_acc': random_acc,
        }
        
        experiment_results.append(exp_res)

    experiment_results = pd.DataFrame(experiment_results)
    experiment_results.to_pickle(experiment_path / 'metrics.pkl')
    log(f"✓ Result: mean AUC: {round(experiment_results['auc'].mean(), 2)}, "
        f"mean ACC: {round(experiment_results['acc'].mean(), 2)} (random={round(experiment_results['random_acc'].mean(), 2)})", indent=indent+2)
