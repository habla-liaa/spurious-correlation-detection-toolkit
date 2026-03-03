from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from src.utils import save_pickle, load_pickle, log
import numpy as np


def load_or_create_splits(dataset_path, aligns_path, params, indent=0, cache=True):
    dataset_path = Path(dataset_path)
    
    n_splits = params['folds_amount']
    group_column = params['group_column']
    repetitions = params['repetitions']
    split_filename = f"splits-tvt-k_{n_splits}-s_{group_column}-r_{repetitions}.pkl"
    all_splits = load_pickle(aligns_path / split_filename, cache=cache)
    
    log(f"SPLITS {'[COMPUTED]' if all_splits is not None else ''}: {n_splits} folds x {params['repetitions']} repetitions (group: {group_column})", indent=indent)
    if all_splits is None:
        sample_ids = load_pickle(aligns_path / 'aligns.pkl', cache=True).sample_id.unique()
        metadata = load_pickle(dataset_path / 'metadata.pkl', cache=True)    
        metadata = metadata[metadata.sample_id.isin(sample_ids)]
        subset = metadata.groupby(group_column).first()        

        group_ids = subset.index.to_numpy()
        conditions = subset['condition']
        
        all_splits = []
        for seed in range(repetitions):
            rskf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed) 
            splits = list(rskf.split(group_ids, conditions))
            fold_splits = []
            for i in range(n_splits):
                test = splits[i][1]
                test_samples = metadata[metadata[group_column].isin(group_ids[test])].sample_id.unique().tolist()
                val = splits[(i + 1) % n_splits][1]
                val_samples = metadata[metadata[group_column].isin(group_ids[val])].sample_id.unique().tolist()
                not_train = np.concatenate([test, val])
                train_samples = metadata[~metadata[group_column].isin(group_ids[not_train])].sample_id.unique().tolist()                            
                fold_splits.append((train_samples, val_samples, test_samples))
            all_splits.append(fold_splits)

        save_pickle(all_splits, aligns_path / split_filename)
    
    return aligns_path / split_filename
