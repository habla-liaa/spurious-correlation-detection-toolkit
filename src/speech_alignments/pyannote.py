from pathlib import Path
from random import randint

import pandas as pd
import tqdm


from src.utils import log, resolve_audio_path, load_audio


def get_name(**kwargs):
	name = f"pyannote"
	if not kwargs.get('postprocess'):
		if kwargs.get('audio_path'):
			name += f"-dataset_{Path(kwargs.get('audio_path')).name}"
	return name


def get_alignments(metadata, params):

	metadata = metadata.copy()
	if params.get('audio_path'):
		log("Resolving audio paths for pyannote aligner...", indent=4)
		metadata['vad_audio_path'] = metadata['file'].apply(lambda path: resolve_audio_path(path, params['audio_path']))
	else:
		metadata['vad_audio_path'] = metadata['file']

	has_limits = 'start' in metadata.columns and 'end' in metadata.columns
	all_speech_segments = []
	
	from pyannote.audio import Pipeline
	pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token=params['huggingface_token'])
	temp_name = f'temporal_alignments_{randint(0, 100)}.pkl'
	temporal_alignments = pd.read_pickle(temp_name) if Path(temp_name).exists() else None
	for (vad_audio_path, sample_id, file), audio_segments in tqdm.tqdm(metadata.groupby(['vad_audio_path', 'sample_id', 'file']), 
																		total=len(metadata.vad_audio_path.unique())):
		
		if not Path(vad_audio_path).exists():
			raise FileNotFoundError(f"Audio file {vad_audio_path} not found.")
		
		sample_rate = 16000
		if temporal_alignments is not None and sample_id in temporal_alignments.sample_id.unique():
			print(f"Using temporal alignments for sample_id {sample_id}...")
			all_speech_segments.append(temporal_alignments[temporal_alignments.sample_id == sample_id])
			continue
		elif has_limits:
			full_waveform = load_audio(vad_audio_path, sample_rate=sample_rate, torch_format=True, torch_params={'normalize':True})
			for _, segment in audio_segments.iterrows():
				if segment['end'] - segment['start'] < 0.5:
					continue  # Skip segments shorter than 0.5 seconds
				segment_waveform = full_waveform[:, int(float(segment['start']) * sample_rate):int(float(segment['end']) * sample_rate)]
				output = pipeline({"waveform": segment_waveform, "sample_rate": sample_rate})
				segments = []
				for (start, end), _ in output.speaker_diarization:
					segments.append([start, end])
				segments = pd.DataFrame(segments, columns=['start', 'end'])
				segments['sample_id'] = sample_id
				segments['file'] = file
				all_speech_segments.append(segments)
		else:
			try:
				output = pipeline({"audio": vad_audio_path, "sample_rate": sample_rate})
			except Exception as e:
				full_waveform = load_audio(vad_audio_path, sample_rate=sample_rate, torch_format=True, torch_params={'normalize':True})
				output = pipeline({"waveform": full_waveform, "sample_rate": sample_rate})
			segments = []
			for (start, end), _ in output.speaker_diarization:
				segments.append([start, end])
			segments = pd.DataFrame(segments, columns=['start', 'end'])
			segments['sample_id'] = sample_id
			segments['file'] = file
			all_speech_segments.append(segments)
		temporal_alignments = pd.concat(all_speech_segments, ignore_index=True)
		temporal_alignments.to_pickle(temp_name)
	

	if len(all_speech_segments) == 0:
		raise ValueError("No speech segments were detected in the provided audio files using the pyannote VAD model.")
	Path(temp_name).unlink(missing_ok=True) 
	return pd.concat(all_speech_segments, ignore_index=True)
