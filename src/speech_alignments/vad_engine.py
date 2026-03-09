import importlib
import re

import pandas as pd

from src.utils import save_pickle, load_pickle, log, load_samples_to_filter
from src.speech_alignments.without_speech_intervals import get_intervals_without_speech
from src.speech_alignments import clip


def clean_aligns(aligns, experiment_path, save_columns=None, minimum_duration=0.0):
    if save_columns is None:
        save_columns = ['file', 'sample_id', 'interval', 'start', 'end']

    aligns['interval'] = aligns.groupby('sample_id').cumcount()
    aligns['start'] = aligns['start'].round(2)
    aligns['end'] = aligns['end'].round(2)
    aligns['duration'] = aligns['end'] - aligns['start']

    aligns = aligns[aligns.duration >= minimum_duration]
    aligns = aligns[save_columns]
    save_pickle(aligns, experiment_path, 'aligns')
    return aligns


def keep_only_samples_in_all_alignments(alignment_runs):
    common_samples = set.intersection(
        *(set(alignments.sample_id) for alignments in alignment_runs.values())
    )

    for run_path, alignments_df in alignment_runs.items():
        if len(common_samples) != len(alignments_df.sample_id.unique()):
            log(f"Keeping only {len(common_samples)} samples in aligns at {run_path}")
        alignments_df = alignments_df[alignments_df.sample_id.isin(common_samples)]
        save_pickle(alignments_df, run_path, 'aligns')


def load_or_create_alignments(experiment_path, aligner_name, params, indent=0, cache=True):
    metadata = load_pickle(experiment_path / 'metadata.pkl', cache=True)
    
    aligner_module_path = f"src.speech_alignments.{aligner_name}"
    try:
        aligner_module = importlib.import_module(aligner_module_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"Module {aligner_module_path} error: {exc}") from exc
    
    if params.get('samples_to_ignore'):
        samples_to_ignore, filter_name = load_samples_to_filter(params.get('samples_to_ignore'), subset=params.get('subset', None))
        metadata = metadata[~metadata.sample_id.isin(samples_to_ignore)]
        experiment_path = experiment_path / f"filter-{filter_name}"
        save_pickle(metadata, experiment_path, "metadata")
        log(f"Removed {len(samples_to_ignore)} samples from non-speech aligns based on {filter_name}", indent=indent + 2)
    
    ######## SPEECH SEGMENTS ########
    base_name = getattr(aligner_module, "get_name")(**params)
    speech_aligns_path = experiment_path / f"aligner-{base_name}"
    speech_alignments = load_pickle(speech_aligns_path / "aligns.pkl", cache=cache)
    log(f"SPEECH ALIGNMENTS {'[COMPUTED]' if speech_alignments is not None else ''}: {base_name.lower()}", indent=indent)
    if speech_alignments is None:
        speech_alignments = getattr(aligner_module, "get_alignments")(metadata, params)
        speech_alignments = clean_aligns(speech_alignments, speech_aligns_path)
    log(f"✓ Result: {speech_alignments.sample_id.nunique()} samples", indent=indent + 2)
    alignment_runs = {speech_aligns_path: speech_alignments}

    ######## NON-SPEECH SEGMENTS ########
    if not re.search(r'(?:^full)', aligner_name):
        without_speech_aligns_path = speech_aligns_path.parent / f"{speech_aligns_path.name}-non_speech"
        without_speech_aligns = load_pickle(without_speech_aligns_path / "aligns.pkl", cache=cache)
        log(f"NON-SPEECH ALIGNMENTS {'[COMPUTED]' if without_speech_aligns is not None else ''}: {without_speech_aligns_path.name}", indent=indent)
        if without_speech_aligns is None:
            unaligned_metadata = metadata[~metadata.sample_id.isin(speech_alignments.sample_id)] if speech_alignments is not None else metadata
            without_speech_aligns = get_intervals_without_speech(speech_alignments, unaligned_metadata)
            without_speech_aligns = clean_aligns(without_speech_aligns, without_speech_aligns_path, minimum_duration=0.5)
        log(f"✓ Result: {without_speech_aligns.sample_id.nunique()} samples", indent=indent+2)
        alignment_runs[without_speech_aligns_path] = without_speech_aligns
       
        if params.get('postprocess'):
            postprocess_cfg = params['postprocess']
            aligner_module_path = f"src.speech_alignments.{postprocess_cfg['name']}"
            try:
                aligner_module = importlib.import_module(aligner_module_path)
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(f"Module {aligner_module_path} error: {exc}") from exc
            postprocess_cfg["params"]['postprocess'] = True
            postprocess_name = getattr(aligner_module, "get_name")(**postprocess_cfg["params"])
            postprocess_without_speech_aligns_path = experiment_path / f"{speech_aligns_path.name}-postprocess-{postprocess_name}-non_speech"
            postprocess_without_speech_aligns = load_pickle(postprocess_without_speech_aligns_path / "aligns.pkl", cache=cache)
            log(f"ALIGNMENTS WITH POSTPROCESS {'[COMPUTED]' if postprocess_without_speech_aligns is not None else ''}: {postprocess_name.lower()}", indent=indent)
            if postprocess_without_speech_aligns is None:
                postprocess_params = postprocess_cfg["params"]
                postprocess_params["audio_path"] = params.get("audio_path", None)
                postprocess_speech_aligns = getattr(aligner_module, "get_alignments")(without_speech_aligns, postprocess_params)
                speech_alignments = pd.concat([speech_alignments, postprocess_speech_aligns])
                
                postprocess_without_speech_aligns = get_intervals_without_speech(speech_alignments)
                postprocess_without_speech_aligns = clean_aligns(postprocess_without_speech_aligns, postprocess_without_speech_aligns_path, minimum_duration=0.5)
            
            log(f"✓ Result: {postprocess_without_speech_aligns.sample_id.nunique()} samples", indent=indent+2)
            del alignment_runs[without_speech_aligns_path]
            alignment_runs[postprocess_without_speech_aligns_path] = postprocess_without_speech_aligns
    
    ####### CLIP ALIGNMENTS #######
    if params.get('clip'):
        name = clip.get_name(params['clip'])
        align_paths = list(alignment_runs.keys())
        for align_path in align_paths:
            clip_aligns_path = align_path / name
            clipmed_alignments = load_pickle(clip_aligns_path / "aligns.pkl", cache=cache)
            log(f"CLIPMED ALIGNMENTS {'[COMPUTED]' if clipmed_alignments is not None else ''}: {name.lower()}", indent=indent)
            if clipmed_alignments is None:
                clipmed_alignments = clip.clip_alignment(alignment_runs[align_path], params['clip'])
                save_pickle(clipmed_alignments, clip_aligns_path, "aligns")
            log(f"✓ Result: {clipmed_alignments.sample_id.nunique()} samples.", indent=indent+2)
            del alignment_runs[align_path]
            alignment_runs[clip_aligns_path] = clipmed_alignments

    return list(alignment_runs.keys())
