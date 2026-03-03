from pathlib import Path

import pandas as pd
import torchaudio
import tqdm

from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import Vad
from src.utils import resolve_audio_path


def get_name(params, postprocess=False):
    name = "torchvad"
    if not postprocess:
        if params.get('audio_path'):
            name += f"-dataset_{Path(params.get('audio_path')).name}"
    return name


def get_alignments(metadata, params):
    speech_windows_df = []
    
    if params.get('audio_path'):
        metadata['vad_audio_path'] = metadata['file'].apply(lambda path: resolve_audio_path(path, params['audio_path']))
    else:
        metadata['vad_audio_path'] = metadata['file']
    
    
    for vad_audio_path, audio_segments in tqdm.tqdm(metadata.groupby('vad_audio_path'), total=len(metadata.vad_audio_path.unique())):
        if not Path(vad_audio_path).exists():
            raise FileNotFoundError(f"Audio file {vad_audio_path} not found.")
        
        audio_sr = torchaudio.info(vad_audio_path).sample_rate

        for _, segment in audio_segments.iterrows():
            if segment['end'] - segment['start'] < 0.5:
                continue  # Skip segments shorter than 0.5 seconds
            
            start_sample = int(float(segment['start']) * audio_sr)
            end_sample = int(float(segment['end']) * audio_sr)
            waveform, sample_rate = torchaudio.load(vad_audio_path, normalize=True, frame_offset=start_sample, num_frames=end_sample - start_sample)
            assert sample_rate == audio_sr, "Sample rate mismatch!"
            transform = Vad(sample_rate=sample_rate, **params.get('torchVAD_params', {}))   
            
            waveform_front_clip = transform(waveform)
            waveform_reversed, sample_rate = apply_effects_tensor(waveform, sample_rate, [["reverse"]])
            assert waveform.size(1) == waveform_reversed.size(1), "Reversed waveform size mismatch!"
            waveform_end_clip = transform(waveform_reversed)
            
            if waveform_end_clip.size(1) == 0 and waveform_front_clip.size(1) == 0:
                continue  # No speech detected

            if waveform_end_clip.size(1) != waveform.size(1): 
                end_sample = end_sample - (waveform_reversed.size(1) - waveform_end_clip.size(1))

            if waveform_front_clip.size(1) < waveform.size(1): 
                start_sample = start_sample + (waveform.size(1) - waveform_front_clip.size(1))
            
            speech_windows_df.append({
                'start': start_sample / sample_rate,
                'end': end_sample / sample_rate,
                'file': segment['file'],
                'sample_id': segment['sample_id']
            })    
    
    return pd.DataFrame(speech_windows_df)