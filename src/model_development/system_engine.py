import pandas as pd
import torch
import tqdm
import gc
from joblib import Parallel, delayed
from src.utils import save_pickle, load_pickle, log
from src.model_development.train import get_early_stopping_name, train
from src.model_development.evaluate import test
from src.model_development.dataset import SpuriousCorrelationDataset
from src.model_development.metrics import compute_metrics
from src.model_development.bootstrapping import compute_bootstrapping, get_name as get_boot_name
import numpy as np


def process_fold(dataset, params, fold_index, train_ids, val_ids, test_ids, experiment_path, cache=True):
    experiment_filename = experiment_path / f"scores-test-f_{fold_index}.pkl"
    if cache and experiment_filename.exists():
        return
    
    current_params = params.copy()
    current_params['model_device'] = 'cuda' if params.get('device', 'gpu') == 'gpu' and torch.cuda.is_available() else 'cpu'
    
    train_dataset = SpuriousCorrelationDataset(dataset, current_params, 'train', train_ids)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=True)
        
    val_dataset = SpuriousCorrelationDataset(dataset, current_params, 'val', val_ids)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=None, shuffle=True)
    
    in_channels = train_dataset.in_channel_size()
    num_classes = train_dataset.amount_classes()
    loss_tracker = []
    if current_params.get('early_stopping'):
        _, loss_tracker, epochs = train(train_loader, val_loader, in_channels, num_classes, current_params)
        current_params['epochs'] = epochs
        current_params['early_stopping'] = False
    
    trainval_dataset = SpuriousCorrelationDataset(dataset, current_params, 'train', np.concatenate([train_ids, val_ids]))
    trainval_loader = torch.utils.data.DataLoader(trainval_dataset, batch_size=None, shuffle=True)
    trained_model, final_loss_tracker, _ = train(trainval_loader, None, in_channels, num_classes, current_params)
    final_loss_tracker['split'] = 'train+val'
    if loss_tracker is not None:
        final_loss_tracker = pd.concat([loss_tracker, final_loss_tracker], ignore_index=True).reset_index(drop=True)
    save_pickle(final_loss_tracker, experiment_path / f"loss_tracker-f_{fold_index}.pkl")

    test_dataset = SpuriousCorrelationDataset(dataset, current_params, 'test', test_ids)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, shuffle=False)
    results = test(trained_model, test_loader)    
    save_pickle(results, experiment_filename)

    del trained_model, train_loader, test_loader, train_dataset, test_dataset
    if current_params['model_device'] == 'cuda':
        torch.cuda.empty_cache()    
    gc.collect()


def get_name(params):
    name = f"model-{params['name']}"
    for c, v in params.items():
        if c not in ['name', 'device', 'n_jobs', 'shuffle', 'drop', 'wrap', 'bootstrapping', 'early_stopping']:
            name += f"-{c}_{v}"
    
    name += f"-{'shuffle' if params.get('shuffle', True) else 'wo_shuffle'}"
    name += "-drop" if params.get("drop_last") else ''
    name += "-wrap" if params.get("wrap_last") else ''
    name += get_early_stopping_name(params)
    return name


def system_development(metadata_path, features_path, splits_path, model_parameters, indent=0, cache=True):
    metadata = load_pickle(metadata_path / 'metadata.pkl', cache=True)
    features_df = load_pickle(features_path / 'features.pkl', cache=True)
    
    splits = load_pickle(splits_path, cache=True)
    dataset = pd.merge(features_df, metadata, on='sample_id')
    dataset = dataset.set_index('sample_id')
    
    experiment_name = get_name(model_parameters)
    output_path = features_path / experiment_name
    
    metrics = load_pickle(output_path / 'metrics.pkl', cache=cache)
    log(f"EXPERIMENT {'[COMPUTED]' if metrics is not None else ''}: {experiment_name.lower()}", indent=indent)
    if metrics is None:
        runtime_model_params = model_parameters.copy()
        for repetition_index, repetition_splits in tqdm.tqdm(enumerate(splits), total=len(splits)):
            repetition_dir = output_path / f"seed_{repetition_index}"
            repetition_dir.mkdir(parents=True, exist_ok=True)
            
            Parallel(n_jobs=runtime_model_params.get('n_jobs', -1))(
                delayed(process_fold)(dataset, runtime_model_params, 
                                      fold_index, train_ids, val_ids, test_ids, repetition_dir, cache=cache)
                for fold_index, (train_ids, val_ids, test_ids) in enumerate(repetition_splits)
            )
    compute_metrics(output_path, indent=indent)
    
    if model_parameters.get('bootstrapping'):
        boot_name = get_boot_name(model_parameters['bootstrapping'])
        bootstrap = load_pickle(output_path / f'{boot_name}.pkl', cache=cache)
        log(f"BOOTSTRAPPING {'[COMPUTED]' if bootstrap is not None else ''}: {boot_name.lower()}", indent=indent)
        if bootstrap is None:
            compute_bootstrapping(output_path, model_parameters['bootstrapping'])
    
    return output_path
