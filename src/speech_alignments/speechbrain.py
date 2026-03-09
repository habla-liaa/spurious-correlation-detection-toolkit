from pathlib import Path
import torchaudio

import pandas as pd
import torch
import tqdm
from speechbrain.inference.VAD import VAD
from src.utils import resolve_audio_path, load_audio


def get_name(**kwargs):
	name = "speechbrain"
	if not kwargs.get('postprocess'):
		if kwargs.get('audio_path'):
			name += f"-dataset_{Path(kwargs.get('audio_path')).name}"
	return name


def get_target_sample_rate(vad_model, params):
	if params.get('sample_rate'):
		return int(params['sample_rate'])

	hparams = getattr(vad_model, 'hparams', None)
	if hparams is not None:
		if isinstance(hparams, dict) and hparams.get('sample_rate'):
			return int(hparams['sample_rate'])

		if not getattr(hparams, 'sample_rate'):
			return int(hparams.sample_rate)

	return 16000


def get_alignments(metadata, params):
	metadata = metadata.copy()
	if params.get('audio_path'):
		metadata['vad_audio_path'] = metadata['file'].apply(lambda path: resolve_audio_path(path, params['audio_path']))
	else:
		metadata['vad_audio_path'] = metadata['file']

	has_limits = 'start' in metadata.columns and 'end' in metadata.columns
	all_speech_segments = []
	
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	vad_model = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty", run_opts={"device": device})
	target_sample_rate = get_target_sample_rate(vad_model, params)

	try:
		for (vad_audio_path, sample_id, file), audio_segments in tqdm.tqdm(metadata.groupby(['vad_audio_path', 'sample_id', 'file']), 
																		total=len(metadata.vad_audio_path.unique())):
			
			if not Path(vad_audio_path).exists():
				raise FileNotFoundError(f"Audio file {vad_audio_path} not found.")
			
			full_waveform = load_audio(vad_audio_path, sample_rate=target_sample_rate, torch_format=True, torch_params={'normalize':True})
			
			if has_limits:
				for _, segment in audio_segments.iterrows():
					if segment['end'] - segment['start'] < 0.5:
						continue  # Skip segments shorter than 0.5 seconds
					start_sample = int(float(segment['start']) * target_sample_rate)
					end_sample = int(float(segment['end']) * target_sample_rate)
					segment_waveform = full_waveform[:, start_sample:end_sample]
					torchaudio.save('temporal_waveform.wav', segment_waveform, target_sample_rate)
					boundaries = vad_model.get_speech_segments('temporal_waveform.wav').cpu().numpy()
					
					boundaries_df = pd.DataFrame(boundaries, columns=['start', 'end'])
					boundaries_df['start'] += segment['start']
					boundaries_df['end'] += segment['start']
					boundaries_df['sample_id'] = sample_id
					boundaries_df['file'] = file
					all_speech_segments.append(boundaries_df)
				
			else:
				torchaudio.save('temporal_waveform.wav', full_waveform, target_sample_rate)
				boundaries = vad_model.get_speech_segments('temporal_waveform.wav').cpu().numpy()			
				boundaries_df = pd.DataFrame(boundaries, columns=['start', 'end'])
				boundaries_df['sample_id'] = sample_id
				boundaries_df['file'] = file
				all_speech_segments.append(boundaries_df)
	finally:
		Path('temporal_waveform.wav').unlink(missing_ok=True)  # Clean up the temporary file
	
	if len(all_speech_segments) == 0:
		raise ValueError("No speech segments were detected in the provided audio files using the SpeechBrain VAD model.")
	return pd.concat(all_speech_segments, ignore_index=True)
