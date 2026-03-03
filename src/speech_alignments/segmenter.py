import pandas as pd
import numpy as np

def get_name(params):
    name = 'presegmenter'
    name += f"-size_{params['size']}-overlap_{params.get('overlap', 0)}"
    return name


def segmenter_alignment(aligns, params):
    segment_size = params['size']
    segment_overlap = params.get('overlap', 0)

    new_aligns = []
    aligns['duration'] = aligns['end'] - aligns['start']
    
    for _, segment in aligns.iterrows():
        for start in np.arange(segment['start'], segment['end'], segment_size - segment_overlap):
            end = start + segment_size
            if end > segment['end']:
                continue
            new_aligns.append({
                'file': segment['file'],
                'sample_id': segment['sample_id'],
                'start': round(start, 2),
                'end': round(end, 2)
            })
    
    new_aligns = pd.DataFrame(new_aligns)
    new_aligns['duration'] = (new_aligns['end'] - new_aligns['start']).round(2)
    assert all(new_aligns['duration']==segment_size), "Error at pre-segmentation."
    return new_aligns.drop(columns=['duration'])
