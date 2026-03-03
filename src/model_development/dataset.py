import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils import log


class SpuriousCorrelationDataset(Dataset):
    def __init__(self, dataset, params, status, sampler):
        self.group_column_name = params['group_column']
        
        self.ids = dataset.loc[sampler].index.to_numpy()
        self.embeddings = dataset.loc[sampler, 'features'].to_numpy()
        self.labels = dataset.loc[sampler, 'condition'].to_numpy()
        self.group_column = dataset.loc[sampler, self.group_column_name].to_numpy()

        if params.get('shuffle', True) and status == 'train':
            shuffle_indices = np.arange(len(self.ids))
            np.random.shuffle(shuffle_indices)
            self.ids = self.ids[shuffle_indices]
            self.embeddings = self.embeddings[shuffle_indices]
            self.labels = self.labels[shuffle_indices]
            self.group_column = self.group_column[shuffle_indices]

        self.batch_size = params['batch_size']
        self.device = params['model_device']

        self.drop_last = params.get("drop_last", False) if status == 'train' else False
        self.wrap_last = params.get("wrap_last", False) if status == 'train' else False

        if self.drop_last and self.wrap_last:
            raise ValueError("Choose either drop_last or wrap_last, not both.")

    def __len__(self):
        if self.drop_last:
            return len(self.ids) // self.batch_size
        return (len(self.ids) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        n = len(self.ids)
        start = index * self.batch_size
        end = start + self.batch_size

        if start >= n:
            raise IndexError

        if end <= n:
            idxs = np.arange(start, end)
        else:
            if self.drop_last:
                raise IndexError
            if self.wrap_last:
                overflow = end - n
                idxs = np.concatenate([np.arange(start, n), np.arange(0, overflow)])
            else:
                idxs = np.arange(start, n)
        
        batch_ids = self.ids[idxs]
        batch_features = self.embeddings[idxs]
        batch_labels = self.labels[idxs]
        batch_groups = self.group_column[idxs]
        
        try:
            X = np.stack(batch_features, axis=0)
        except ValueError:                
            max_len = max([x.shape[0] for x in batch_features]) 
            feature_dim = batch_features[0].shape[1]
            padded_X = np.zeros((len(batch_features), max_len, feature_dim))
            for i, sample in enumerate(batch_features):
                length = sample.shape[0]
                padded_X[i, :length, :] = sample
            X = padded_X
            
        X = np.transpose(X, (0, 2, 1))
            
        return {'features': torch.from_numpy(X).float().to(self.device),
                'labels': torch.from_numpy(batch_labels).to(self.device),
                'ids': batch_ids, 
                'groups': batch_groups}
    
    def apply_global_normalization(self, normalization_stats):      
        log(" --> Applying global normalization to the dataset", indent=8)

        if normalization_stats is None:
            all_features = np.vstack(self.embeddings)
            self.mean_norm = np.mean(all_features, axis=0).astype(np.float32)
            self.std_norm = np.std(all_features, axis=0).astype(np.float32)
        else:
            self.mean_norm = np.asarray(normalization_stats[0], dtype=np.float32)
            self.std_norm = np.asarray(normalization_stats[1], dtype=np.float32)

        self.std_norm = np.where(np.abs(self.std_norm) < 1e-8, 1.0, self.std_norm).astype(np.float32)

        for i in range(len(self.embeddings)):
            self.embeddings[i] = ((self.embeddings[i] - self.mean_norm) / self.std_norm).astype(np.float32, copy=False)
        
    def global_normalization_values(self):
        return self.mean_norm.tolist(), self.std_norm.tolist()

    def amount_classes(self):
        return len(np.unique(self.labels))

    def in_channel_size(self):
        return self.embeddings[0].shape[1]

    def amount_batches(self):
        return len(self) 
