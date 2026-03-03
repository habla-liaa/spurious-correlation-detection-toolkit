import glob
import joblib
import os
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from pydub import AudioSegment


def log(text, indent=0, newline=False):
    prefix = "\n" if newline else ""
    print(f"{prefix}{' ' * indent}{text}")


def configure_download_cache(cache_dir=None):
    if not cache_dir:
        return None

    cache_root = Path(cache_dir).expanduser()
    huggingface_home = cache_root / 'huggingface'
    huggingface_hub = huggingface_home / 'hub'
    transformers_cache = huggingface_home / 'transformers'
    torch_home = cache_root / 'torch'
    torch_hub = torch_home / 'hub'
    whisper_cache = cache_root / 'whisper'

    for path in [cache_root, huggingface_home, huggingface_hub, transformers_cache, torch_home, torch_hub, whisper_cache]:
        path.mkdir(parents=True, exist_ok=True)

    os.environ['XDG_CACHE_HOME'] = str(cache_root)
    os.environ['HF_HOME'] = str(huggingface_home)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(huggingface_hub)
    os.environ['TRANSFORMERS_CACHE'] = str(transformers_cache)
    os.environ['TORCH_HOME'] = str(torch_home)  
    torch.hub.set_dir(str(torch_hub))

    return {
        'root': str(cache_root),
        'whisper': str(whisper_cache),
        'huggingface': str(huggingface_hub),
        'torch_hub': str(torch_hub),
    }


def get_audio_duration(filename):
    return len(AudioSegment.from_file(filename)) / 1000


def load_pickle(filename, type='pandas', cache=True):
    if Path(filename).exists() and cache:
        if type == 'pandas':
            return pd.read_pickle(filename)
        else:
            return joblib.load(filename)
    return None


def save_pickle(data, experiment_path, filename=None):
    complete_filename = Path(experiment_path)
    if filename is None:
        complete_filename.parent.mkdir(parents=True, exist_ok=True)
    else:
        complete_filename.mkdir(parents=True, exist_ok=True)
        complete_filename = complete_filename / f'{filename}.pkl'

    if isinstance(data, list):
        joblib.dump(data, complete_filename)
    else:                
        data.to_pickle(complete_filename)

    return complete_filename


def resolve_audio_path(original_audio_path, dataset_root):
    candidate_paths = glob.glob(f"{dataset_root}/**/*{Path(original_audio_path).stem}*", recursive=True)
    if not candidate_paths:
        from IPython import embed; embed()  # DEBUG
        raise FileNotFoundError(f"Audio file {original_audio_path} not found under {dataset_root}")
    return candidate_paths[0]
    

def mix_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 1:
        return waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def load_audio(audio_file, sample_rate=16000, torch_format=False, torch_params=None):
    waveform, sr = torchaudio.load(audio_file, **torch_params) if torch_params else torchaudio.load(audio_file)
    waveform = mix_to_mono(waveform).to(torch.float32)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if torch_format:
        return waveform
    return waveform.squeeze(0).cpu().numpy()
    

def load_samples_to_filter(filtered_samples_file, subset=None):
    if filtered_samples_file:
        if subset and '[subset]' in filtered_samples_file:
            filtered_samples_file = filtered_samples_file.replace('[subset]', subset)
        filtered_samples_file = Path(filtered_samples_file)
        if not filtered_samples_file.exists():
            raise FileNotFoundError(f"Samples to ignore file not found: {filtered_samples_file}")
        with open(filtered_samples_file, 'r') as f:
            samples_to_filter = f.read().split(',')
        samples_to_filter = [item.strip() for item in samples_to_filter]
        return samples_to_filter, filtered_samples_file.stem
    return [], 'no_file'
