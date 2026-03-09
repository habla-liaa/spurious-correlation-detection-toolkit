import argparse
import importlib
from pathlib import Path

import yaml

from src.splits import load_or_create_splits
from src.speech_alignments.vad_engine import load_or_create_alignments
from src.dataset_readers.preprocessing import adjust_audio_sample_rates
from src.features.features_engine import load_or_create_audio_representation
from src.model_development.system_engine import system_development
from src.utils import log


def load_dataset(config):
    dataset_cfg = config["dataset"]
    dataset_output_dir = Path(config["experiment_output_dir"]) / f'dataset-{dataset_cfg["name"]}'
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    module_path = f"src.dataset_readers.{dataset_cfg['name']}"
    try:
        dataset_module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"Could not import module '{module_path}': {exc}") from exc

    read_dataset = getattr(dataset_module, "read_dataset")
    subset_names = read_dataset(dataset_output_dir, dataset_cfg)

    return dataset_output_dir, subset_names


def run_pipeline(config):
    dataset_cfg = config["dataset"]
    cache = config.get("cache", False)
    log("CACHE MODE: " + ("ON" if cache else "OFF"), newline=True)

    log("=" * 80, newline=True)
    log(f"START PIPELINE: dataset={dataset_cfg['name']}")
    log("=" * 80)

    dataset_output_dir, subset_names = load_dataset(config)
    config["dataset_names"] = subset_names

    aligners_cfg = config.get("aligners", [])
    features_cfg = config.get("features", [])
    for subset_name in subset_names:
        log("#" * 90, indent=1)
        log(f"SUBSET: {subset_name}", indent=1)
        subset_output_dir = dataset_output_dir / f'subset-{subset_name}'
        subset_output_dir.mkdir(parents=True, exist_ok=True)        
        subset_output_dir = adjust_audio_sample_rates(subset_output_dir, dataset_cfg.get('adjusted_audio_path'))
        for aligner_cfg in aligners_cfg:
            log('PREPROCESSING ALIGNMENTS', indent=2)
            aligner_params = aligner_cfg.get('params', {}).copy()
            aligner_params['experiment_output_dir'] = str(dataset_output_dir)
            aligner_params['subset'] = subset_name
            alignment_dirs = load_or_create_alignments(subset_output_dir, aligner_cfg['name'], aligner_params, indent=2, cache=cache)
            if not features_cfg: 
                log("!! INFO :: No features specified in the configuration. Alignments will be created but no experiments will be run.", indent=2)
                continue
            
            for alignment_dir in alignment_dirs:
                log("-" * 90, indent=2)
                if config.get('only_alignment_contains') and config['only_alignment_contains'] not in str(alignment_dir):
                    log(f'!! INFO :: Skipping alignment dir because {config["only_alignment_contains"]} not found in {alignment_dir.name}.', indent=3)
                    continue
                if config.get('exclude_alignment_contains') and config['exclude_alignment_contains'] in str(alignment_dir):
                    log(f'!! INFO :: Skipping alignment dir because {config["exclude_alignment_contains"]} found in {alignment_dir.name}.', indent=3)
                    continue
                
                log(f"ALIGNMENT: {alignment_dir.name}", indent=2)
                splits_dir = load_or_create_splits(subset_output_dir, alignment_dir, config["splits"], indent=2, cache=cache)
                
                for feature_cfg in features_cfg:
                    
                    log("-" * 90, indent=2)
                    features_dir = load_or_create_audio_representation(alignment_dir, feature_cfg['name'], feature_cfg.get('params', {}),
                                                                       indent=3, cache=cache)
                    experiment_dir = system_development(subset_output_dir, features_dir, splits_dir, config['model'], indent=3, cache=cache)
                  
                    with (experiment_dir / 'config.yaml').open('w', encoding='utf-8') as config_file:
                        yaml.safe_dump(config, config_file)
                    log(f"EXPERIMENT SAVED: {str(experiment_dir).replace(config['experiment_output_dir'], '')}", indent=3)
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cache', dest='cache', action='store_true')
    parser.add_argument('-cf', '--config', dest='config_file', required=True, help='Path to the configuration YAML file')
    parser.add_argument('--download-cache-dir', dest='download_cache_dir', default=None,
                        help='Directory used by external model download caches (whisper, torch hub, huggingface).')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = yaml.safe_load(Path(args.config_file).read_text())
    config['cache'] = args.cache
    if args.download_cache_dir:
        config['download_cache_dir'] = args.download_cache_dir

    run_pipeline(config)
