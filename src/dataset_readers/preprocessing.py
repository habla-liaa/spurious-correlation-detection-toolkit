import torchaudio
from pathlib import Path
from src.utils import load_audio, log, load_pickle, save_pickle


def adjust_audio_sample_rates(subset_experiment_path, output_path=None):
    resample_experiment_path = subset_experiment_path / 'resampled'
    resample_experiment_path.mkdir(parents=True, exist_ok=True)
    if not output_path:
        resample_audios_path = resample_experiment_path / 'audios'
    else:
        resample_audios_path = Path(output_path)
    resample_audios_path.mkdir(parents=True, exist_ok=True)
    
    metadata = load_pickle(subset_experiment_path / 'metadata.pkl', cache=True)
    metadata = metadata.assign(sample_rate=0)
    for i, row in metadata.iterrows():
        _, sr = torchaudio.load(row['file'])
        metadata.at[i, 'sample_rate'] = sr
    
    if len(metadata.sample_rate.unique()) == 0:
        return metadata
    
    min_sample_rate = metadata.sample_rate.min()
    if len(metadata.sample_rate.unique()) > 1:
        log(f'WARNING: Found multiple sample rates in the dataset: {metadata.sample_rate.unique()}. Resampling to the minimum sample rate: {min_sample_rate} Hz.')

    for i, row in metadata.iterrows():
        audio_path = Path(row['file'])
        sample_id = row['sample_id']
        if row['sample_rate'] != min_sample_rate:
            audio_signal = load_audio(audio_path, sample_rate=min_sample_rate, torch_format=True)
            adjusted_audio_file_path = resample_audios_path / f"{sample_id}-sr_{min_sample_rate}{audio_path.suffix}"
            torchaudio.save(adjusted_audio_file_path, audio_signal, min_sample_rate)
            metadata.at[i, 'file'] = str(adjusted_audio_file_path)
        else:
            adjusted_audio_file_path = row['file']
        if min_sample_rate != 16000:
            audio_signal = load_audio(adjusted_audio_file_path, sample_rate=16000, torch_format=True)
            adjusted_audio_file_path = resample_audios_path / f"{sample_id}-sr_16000{audio_path.suffix}"
            torchaudio.save(adjusted_audio_file_path, audio_signal, 16000)
            metadata.at[i, 'file'] = str(adjusted_audio_file_path)

    metadata = metadata.drop(columns=['sample_rate'])
    save_pickle(metadata, resample_experiment_path, 'metadata')
    return resample_experiment_path
