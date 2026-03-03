import importlib

from src.utils import save_pickle, load_pickle, log
from src.features.normalization import normalizer, get_name as get_normalization_name


def load_or_create_audio_representation(experiment_path, feature_name, params, indent=0, cache=True):
    alignments_df = load_pickle(experiment_path / 'aligns.pkl', cache=True)
    
    min_duration = float(params.get('min_duration', 0.05))
    if min_duration > 0:
        alignments_df = alignments_df[(alignments_df['end'] - alignments_df['start']) >= min_duration].reset_index(drop=True)
        log(f"FILTER SEGMENTS: Removed segments shorter than {min_duration:.3f} seconds", indent=indent)
    
    feature_module_path = f"src.features.{feature_name}"
    try:
        feature_module = importlib.import_module(feature_module_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"Module {feature_module_path} error: {exc}") from exc

    feature_run_name = getattr(feature_module, 'get_name')(params)
    feature_dir = experiment_path / f'feature-{feature_run_name}'
    features = load_pickle(feature_dir / 'features.pkl', cache=cache) 
    log(f"AUDIO REPRESENTATION {'[COMPUTED]' if features is not None else ''}: {feature_run_name.lower()}", indent=indent)
    if features is None:
        features = getattr(feature_module, 'get_embeddings')(alignments_df, params)
        save_pickle(features, feature_dir, 'features')

    normalized_dir = feature_dir
    if params.get('normalization'):
        normalization_name = get_normalization_name(params['normalization'])
        normalized_dir = feature_dir / normalization_name
        normalized_file = normalized_dir / 'features.pkl'
        normalized_features = load_pickle(normalized_file, cache=cache)
        log(f"NORMALIZATION {'[COMPUTED]' if normalized_features is not None else ''}: {normalization_name.lower()}", indent=indent)
        if normalized_features is None:
            features = normalizer(features, params['normalization'])
            save_pickle(features, normalized_dir, 'features')
        else:
            features = normalized_features
    
    return normalized_dir
