from pathlib import Path
import tqdm
import torch
import pandas as pd
from src.utils import log, resolve_audio_path


def get_name(**kwargs):
    name = f"silero-{kwargs.get('version','last')}-t_{kwargs.get('threshold', 0.2)}-sr_{kwargs.get('sample_rate', 16000)}"
    
    if not kwargs.get('postprocess'):
        if kwargs.get('audio_path'):
            name += f"-dataset_{Path(kwargs.get('audio_path')).name}" 
    
    return name


def get_alignments(metadata, params):
    all_speech_segment = []
    if params.get('version', 'last') == 'local':
        model, utils = torch.hub.load(repo_or_dir=params['local_repo_path'], model='silero_vad', source='local', onnx=False)
    else:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=False)
                
    (get_speech_timestamps, _, read_audio, _, _) = utils
    
    if params.get('audio_path'):
        log("Resolving audio paths for silero aligner...", indent=4)
        metadata['vad_audio_path'] = metadata['file'].apply(lambda path: resolve_audio_path(path, params['audio_path']))
    else:
        metadata['vad_audio_path'] = metadata['file']
    
    sample_rate = params.get('sample_rate', 16000)
    has_limits = 'start' in metadata.columns and 'end' in metadata.columns
    threshold = params.get('threshold', 0.2)
    print(f"Silero VAD: Using threshold: {threshold}, sampling rate: {sample_rate}, has_limits: {has_limits}")
    for vad_audio_path, audio_segments in tqdm.tqdm(metadata.groupby('vad_audio_path'), total=len(metadata.vad_audio_path.unique())):
        if not Path(vad_audio_path).exists():
            raise FileNotFoundError(f"Audio file {vad_audio_path} not found.")
        waveform = read_audio(vad_audio_path, sampling_rate=sample_rate)
        
        for _, segment in audio_segments.iterrows():
            if has_limits:
                start_sample = int(float(segment['start']) * sample_rate)
                end_sample = int(float(segment['end']) * sample_rate)
                segment_waveform = waveform[start_sample:end_sample]
            else:
                segment_waveform = waveform
            
            speech_windows_df = pd.DataFrame(get_speech_timestamps(segment_waveform, model, sampling_rate=sample_rate, threshold=threshold))
            
            if len(speech_windows_df) == 0:
                if not has_limits:
                    print('WARNING:: Audio without any speech: ', vad_audio_path)
            else:
                speech_windows_df['start'] = speech_windows_df['start'] / sample_rate + segment.get('start', 0)
                speech_windows_df['end'] = speech_windows_df['end'] / sample_rate + segment.get('start', 0)
                speech_windows_df['file'] = segment['file']
                speech_windows_df['sample_id'] = segment['sample_id']
                all_speech_segment.append(speech_windows_df)
    
    if len(all_speech_segment) == 0:
        raise ValueError("WARNING:: No speech segments detected in any audio file using Silero VAD.")
        
    return pd.concat(all_speech_segment, ignore_index=True)
