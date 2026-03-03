import gc
from pathlib import Path
import tqdm
import torch
import pandas as pd
import whisper
from src.utils import resolve_audio_path

def get_name(params, postprocess=False):
    name = f"whisper-{params['model']}"
    if not postprocess:
        if params.get('audio_path'):
            return f"{name}-dataset_{Path(params.get('audio_path')).name}"
    return name


def normalize_text(t):
    return (t or "").strip().replace("…", "").replace(".", "")


def is_valid_speech_segment(segment, no_speech_prob_th, compress_ratio_th):
    text = segment.get('text', '')
    no_speech_prob = segment.get('no_speech_prob', 0.0)
    compression_ratio = segment.get('compression_ratio', 0.0)

    if not normalize_text(text):
        return False
    
    if no_speech_prob >= no_speech_prob_th:
        return False
    
    if compression_ratio >= compress_ratio_th:
        return False

    return True


def get_alignments(metadata, params):
    all_speech_segments = []

    if params.get("audio_path"):
        metadata["vad_audio_path"] = metadata["file"].apply(lambda p: resolve_audio_path(p, params["audio_path"]))
    else:
        metadata["vad_audio_path"] = metadata["file"]

    sample_rate = int(params.get("sample_rate", 16000))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model(params['model'], device=device)

    no_speech_prob_th = float(params.get("no_speech_prob_threshold", 0.5)) 
    compress_ratio_th = float(params.get("compress_ratio_threshold", 2.4))

    has_limits = 'start' in metadata.columns and 'end' in metadata.columns
    
    for vad_audio_path in tqdm.tqdm(metadata.vad_audio_path.unique(), desc="Processing Audio Files"):
        if not Path(vad_audio_path).exists():
            print(f"Warning: Audio file {vad_audio_path} not found. Skipping.")
            continue
        
        full_waveform = whisper.load_audio(vad_audio_path, sr=sample_rate)    
        audio_segments = metadata[metadata.vad_audio_path == vad_audio_path]        
        for _, audio_segment in audio_segments.iterrows():
            if has_limits:
                chunk_start_sec = float(audio_segment['start'])
                chunk_end_sec = float(audio_segment['end'])
                
                start_sample = int(chunk_start_sec * sample_rate)
                end_sample = int(chunk_end_sec * sample_rate)
                
                segment_waveform = full_waveform[start_sample:end_sample]
            else:
                chunk_start_sec = 0.0
                segment_waveform = full_waveform

            if len(segment_waveform) < sample_rate * 0.1: # seconds
                continue

            result = whisper_model.transcribe(
                segment_waveform, 
                language=params['language'], 
                beam_size=params.get('beam_size', 5), 
                temperature=0.0, 
                condition_on_previous_text=False, 
                word_timestamps=False
            )
            
            for seg in result['segments']:
                if is_valid_speech_segment(seg, no_speech_prob_th, compress_ratio_th):
                    all_speech_segments.append({
                        'sample_id': audio_segment['sample_id'],
                        'file': audio_segment['file'],
                        'start': chunk_start_sec + seg["start"], 
                        'end': chunk_start_sec + seg["end"],
                    })
    
    if device == 'cuda': torch.cuda.empty_cache()    
    gc.collect()
    if len(all_speech_segments) == 0:
        raise ValueError("WARNING:: No valid speech segments detected in any audio file using Whisper VAD.")
    return pd.DataFrame(all_speech_segments)
