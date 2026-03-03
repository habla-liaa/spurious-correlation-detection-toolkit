import numpy as np


def get_name(params):
    return f"normalization-{params['type']}"


def normalizer(features, params):   
    if params['type'] == 'audio_level':
        for _, audio_rows in features.groupby('sample_id'):
            all_file_features = np.vstack(audio_rows['features'].to_numpy())
            mean = np.mean(all_file_features, axis=0)
            std = np.std(all_file_features, axis=0)
            
            normalized_features = audio_rows['features'].apply(lambda f: (f - mean) / std)
            features.loc[audio_rows.index, 'features'] = normalized_features
        return features
    
    else:
        raise ValueError(f"Normalization type {params['type']} not recognized")
