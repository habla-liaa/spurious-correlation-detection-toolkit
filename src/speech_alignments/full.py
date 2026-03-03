from src.utils import get_audio_duration


def get_name(params, postprocess=False):
    return 'full_audio'


def get_alignments(metadata, params):
    if 'file' not in metadata.columns:
        raise KeyError("'file' column not found in metadata DataFrame")
    
    aligns = metadata.copy()
    aligns['start'] = 0
    aligns['end'] = aligns['file'].apply(lambda x: get_audio_duration(x))
        
    return aligns
