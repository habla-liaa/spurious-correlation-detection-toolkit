import pandas as pd


def get_name(segmenter_params):
    name = "segmenter"
    if segmenter_params.get('size'):
        name += f"-s_{segmenter_params['size']}"
    name += f"-o_{segmenter_params.get('overlap', 0)}"
    return name

def segmenter(features, segmenter_params, embedding_window_size):
    features = features.reset_index(drop=True)
    new_segments = []

    segmenter_size = segmenter_params['size']
    overlap = segmenter_params.get('overlap', 0)
    frames_per_segment = int(round(segmenter_size / embedding_window_size))
    overlap_frames = int(round(overlap / embedding_window_size))
    stride = frames_per_segment - overlap_frames
    for _, interval in features.iterrows():
        interval_features = interval['features']
        for i in range(0, len(interval_features), stride):
            current_features = interval_features[i:i + frames_per_segment]
            new_segments.append({
                'file': interval['file'], 'sample_id': interval['sample_id'],
                'features': current_features
            })

    return pd.DataFrame(new_segments, columns=['file', 'sample_id', 'features'])


def calculate_segmenter_params(params, window_size):
    segmenter_size = params['size']
    overlap = params.get('overlap', 0)
    frames_per_segment = int(round(segmenter_size / window_size))
    overlap_frames = int(round(overlap / window_size))
    stride = frames_per_segment - overlap_frames
    return stride, frames_per_segment


def segmenter_by_audio(features, stride, frames):
    new_segments = []
    for i in range(0, len(features), stride):
        new_segments.append(features[i:i + frames])
    return new_segments