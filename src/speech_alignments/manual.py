from pathlib import Path
import pandas as pd
from src.utils import load_pickle


def concatenate_aligns(aligns):
    new_aligns = []
    for _, df in aligns.groupby('sample_id'):
        prev_end = -1
        for _, seg in df.sort_values('start').iterrows():
            content = seg.get('content', None)
            if prev_end != -1 and content != 'SIL' and prev_end == round(seg.start, 2):
                new_aligns[-1]['end'] = seg.end
            else:
                new_aligns.append(seg.to_dict())
            prev_end = round(seg.end, 2)
    return pd.DataFrame(new_aligns)


def get_name(**kwargs):
    return 'manual-' + '_'.join(Path(kwargs['filename']).stem.split('_')[1:])


def get_alignments(metadata, params):
    aligns = load_pickle(Path(params['experiment_output_dir'], 'manual-aligns') / f'{params["filename"]}.pkl', cache=True)
    
    if aligns is not None:
        if {'sample_id', 'start', 'end'}.issubset(aligns.columns):
            aligns = concatenate_aligns(aligns)
        aligns = aligns[aligns.sample_id.isin(metadata.sample_id)]
        aligns = aligns.merge(metadata[['sample_id', 'file']], on='sample_id', how='left')
        return aligns
    else:
        raise FileNotFoundError(f"Manual aligns file {params['filename']} not found.")
