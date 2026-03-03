import torchaudio
import torch
import tqdm
import pandas as pd
import numpy as np

from src.features.segmentation import calculate_segmenter_params
from src.utils import load_audio


def get_spectrogram_torch(audio_signal, extractor):
    noise_feature = extractor(audio_signal)
    return noise_feature.squeeze(0).transpose(0, 1).contiguous().numpy().astype('float32')


def get_name(params):
    name = "spectrogram"
    if params.get('concatenate_audio_segments'):
        name += "-concatAudioSegs"
    elif params.get('concatenate_segments'):
        name += "-concatSegs"
    
    if params.get('segmenter'):
        name += f"-segSize_{params['segmenter']['size']}-segOverlap_{params['segmenter'].get('overlap', 0)}"
    
    for c, v in params.items():
        if c not in ['concatenate_segments', 'segmenter']:
            name += f"-{c}_{v}"
    return name


def get_embeddings(alignment_df, params):
    features = []
    alignment_df = alignment_df.reset_index(drop=True)
    
    extractor = torchaudio.transforms.Spectrogram(
        center=False, pad=0,               
        n_fft=params.get("n_fft", 2048),
        hop_length=params.get("hop_length", 512),
        power=params.get("power", 2.0),
    )
    
    for (audio_file, sample_id), segments in tqdm.tqdm(alignment_df.groupby(['file', 'sample_id']), 
                                                       total=len(alignment_df.file.unique())):
        
        sample_rate = params.get('sample_rate', 16000)
        audio_signal = load_audio(audio_file, sample_rate=sample_rate, torch_format=True)

        if params.get('concatenate_audio_segments'):
            concatenated_signal = []
            for segment in segments.itertuples(index=False):
                start_sample = int(round(segment.start * sample_rate))
                end_sample = int(round(segment.end * sample_rate))
                concatenated_signal.append(audio_signal[:, start_sample:end_sample])
            audio_signal = torch.cat(concatenated_signal, dim=1)
            spectrogram_res = get_spectrogram_torch(audio_signal, extractor)
            features.append({'file': audio_file, 'sample_id': sample_id, 'features': spectrogram_res})
        
        elif params.get('concatenate_segments'):
            segments_res = []
            for segment in segments.itertuples(index=False):       
                start_sample = int(round(segment.start * sample_rate))
                end_sample = int(round(segment.end * sample_rate))
                spectrogram_res = get_spectrogram_torch(audio_signal[:, start_sample:end_sample], extractor)
                if spectrogram_res is not None:
                    segments_res.append(spectrogram_res)
            segments_res = np.concatenate(segments_res)
            features.append({
                'file': audio_file, 'sample_id': 
                sample_id, 'features': segments_res
            })
        
        else:
            for segment in segments.itertuples(index=False):      
                start_sample = int(round(segment.start * sample_rate))
                end_sample = int(round(segment.end * sample_rate))
                segment_audio_signal = audio_signal[:, start_sample:end_sample]
                spectrogram_res = get_spectrogram_torch(segment_audio_signal, extractor)

                if spectrogram_res is None: continue
                features.append({
                    'file': audio_file, 'sample_id': sample_id,
                    'features': spectrogram_res 
                })

    features_df = pd.DataFrame(features)
    if params.get('segmenter'):
        stride, frames = calculate_segmenter_params(params['segmenter'], params.get('hop_length', 160) / sample_rate)
        segmented_features = []
        for row in features_df.itertuples(index=False):
            segment_feature = row.features
            for i in range(0, len(segment_feature), stride):
                interval = segment_feature[i:i + frames]
                segmented_features.append({
                    'file': row.file, 'sample_id': row.sample_id,
                    'features': interval,
                })
        features_df = pd.DataFrame(segmented_features)
    return features_df
