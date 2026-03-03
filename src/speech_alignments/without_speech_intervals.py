import pandas as pd

from src.utils import get_audio_duration

def get_intervals_without_speech(aligns, audios_without_speech_aligns=[]):
    if aligns is not None:
        invert_aligns = []
        for _, aligns_for_file in aligns.groupby('sample_id'):
            inv_start = 0
            for _, seg in aligns_for_file.sort_values('start').iterrows():
                seg = seg.to_dict()
                if seg['start'] > inv_start:
                    gap = seg.copy()
                    gap['end'] = seg['start']
                    gap['start'] = inv_start
                    invert_aligns.append(gap)
                inv_start = max(inv_start, seg['end'])
            audio_duration = get_audio_duration(seg['file'])
            if inv_start < audio_duration:
                seg = seg.copy()
                seg['start'] = inv_start
                seg['end'] = audio_duration
                invert_aligns.append(seg)
    else:
        invert_aligns = []
    
    invert_aligns = pd.DataFrame(invert_aligns)
    
    if len(audios_without_speech_aligns) != 0:
        audios_without_speech_aligns['start'] = 0
        audios_without_speech_aligns['end'] = audios_without_speech_aligns.file.apply(lambda file: get_audio_duration(file))
        return pd.concat([invert_aligns, audios_without_speech_aligns])

    return invert_aligns
