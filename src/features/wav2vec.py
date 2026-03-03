import numpy as np
import pandas as pd
import torch
import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2Model

from src.features.segmentation import calculate_segmenter_params
from src.utils import load_audio


def compute_wav2vec_embeddings(audio_segment, feature_extractor, model, device, sample_rate, min_frames=0):
    if audio_segment is None or len(audio_segment) == 0 or len(audio_segment) < min_frames:
        return None
    
    inputs = feature_extractor(audio_segment.astype(np.float32, copy=False), sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        all_layers_by_time = model(**inputs, output_hidden_states=True).hidden_states
    return all_layers_by_time


def load_wav2vec_model(device: torch.device, model_path: str):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    sample_rate = getattr(feature_extractor, 'sampling_rate', None)
    if sample_rate is None:
        raise ValueError(f"Feature extractor for {model_path} does not specify a sample rate.")
    model = Wav2Vec2Model.from_pretrained(model_path, use_safetensors=True, local_files_only=False).to(device)
    model.eval()
    return model, feature_extractor, sample_rate


def get_name(params):
    name = "wav2vec"
    if params.get('concatenate_audio_segments'):
        name += "-concatAudioSegs"
    elif params.get('concatenate_segments'):
        name += "-concatSegs"
        
    if params.get('segmenter'):
        name += f"-segSize_{params['segmenter']['size']}-segOverlap_{params['segmenter'].get('overlap', 0)}"
    return f"{name}-{params['model'].replace('/', '-')}-minFrames_{params.get('minFrames', 0)}"


def get_embeddings(alignment_df, params):
    features_df = alignment_df.copy().reset_index(drop=True)
    features_df['features'] = None
    alignment_df = alignment_df.reset_index(drop=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, feature_extractor, sample_rate = load_wav2vec_model(device, params['model'])

    if params.get('segmenter'):
        stride, frames = calculate_segmenter_params(params['segmenter'], 0.02)

    features = []
    for (audio_file, sample_id), segments in tqdm.tqdm(alignment_df.groupby(['file', 'sample_id']), 
                                                       total=len(alignment_df.file.unique())):
        audio_signal = load_audio(audio_file, sample_rate=sample_rate, torch_format=False)

        if params.get('concatenate_audio_segments'):
            concatenated_signal = []
            for segment in segments.itertuples(index=False):
                start_sample = int(round(segment.start * sample_rate))
                end_sample = int(round(segment.end * sample_rate))
                concatenated_signal.append(audio_signal[start_sample:end_sample])
            audio_signal = np.concatenate(concatenated_signal)
            w2v_res = compute_wav2vec_embeddings(audio_signal, feature_extractor, model,
                                                device, sample_rate, min_frames=params.get('minFrames', 0))
            w2v_res = torch.cat(w2v_res, dim=0)
            if params.get('segmenter'):
                for i in range(0, w2v_res.shape[1], stride):
                    interval = w2v_res[:, i:i + frames].mean(dim=1)
                    features.append({
                        'file': audio_file, 'sample_id': sample_id,
                        'features': interval.cpu().numpy().astype(np.float32),
                    })  
            else:
                w2v_res = w2v_res.mean(dim=1)
                features.append({
                    'file': audio_file, 'sample_id': sample_id,
                    'features': w2v_res.cpu().numpy().astype(np.float32),
                })
        
        elif params.get('concatenate_segments'):
            all_audio_res = []
            for _, segment in segments.iterrows():            
                segment_audio_signal = audio_signal[int(segment.start * sample_rate):int(segment.end * sample_rate)]
                w2v_res = compute_wav2vec_embeddings(segment_audio_signal, feature_extractor, model,
                                                     device, sample_rate, min_frames=params.get('minFrames', 0))
                if w2v_res is not None:
                    all_audio_res.append(w2v_res)

            num_layers = len(all_audio_res[0])
            concat_layers = []
            for layer_idx in range(num_layers):
                layer_chunks = [segment_states[layer_idx] for segment_states in all_audio_res]
                merged_layer = torch.cat(layer_chunks, dim=1)
                concat_layers.append(merged_layer)
            concat_layers = torch.cat(concat_layers, dim=0)
            
            if params.get('segmenter'):
                for i in range(0, concat_layers.shape[1], stride):
                    interval = concat_layers[:, i:i + frames].mean(dim=1)
                    features.append({
                        'file': audio_file, 'sample_id': sample_id,
                        'features': interval.cpu().numpy().astype(np.float32),
                    })  
            else:
                features.append({
                    'file': audio_file, 'sample_id': sample_id,
                    'features': concat_layers.mean(dim=1).cpu().numpy().astype(np.float32),
                })  
        
        else:
            for _, segment in segments.iterrows():            
                segment_audio_signal = audio_signal[int(segment.start * sample_rate):int(segment.end * sample_rate)]
                w2v_res = compute_wav2vec_embeddings(segment_audio_signal, feature_extractor, model, 
                                                    device, sample_rate, min_frames=params.get('minFrames', 0))
                if w2v_res is None: continue
                
                w2v_res = torch.cat(w2v_res, dim=0)
                if params.get('segmenter'):
                    for i in range(0, w2v_res.shape[1], stride):
                        interval = w2v_res[:, i:i + frames].mean(dim=1)
                        features.append({
                            'file': audio_file, 'sample_id': sample_id,
                            'features': interval.cpu().numpy().astype(np.float32),
                        })  
                else:
                    w2v_res = w2v_res.mean(dim=1)
                    features.append({
                        'file': audio_file, 'sample_id': sample_id,
                        'features': w2v_res.cpu().numpy().astype(np.float32),
                    })
    
    features_df = pd.DataFrame(features)
    return features_df
