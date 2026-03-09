import subprocess
from pathlib import Path

import tqdm
from src.utils import load_pickle, save_pickle, log, resolve_audio_path


def start_enhance_process():
    from df.enhance import init_df
    model, df_state, _ = init_df()
    return model, df_state


def enhance_single_audio(original_file, enhanced_audio_path, model, df_state, indent):
    from df.enhance import enhance, load_audio, save_audio

    temp_mono_file = enhanced_audio_path.with_name(f"._mono_{enhanced_audio_path.stem}{enhanced_audio_path.suffix}")
    temp_norm_file = enhanced_audio_path.with_name(f"._norm_{enhanced_audio_path.stem}.wav")

    try:
        to_mono = ["ffmpeg", "-y", "-i", str(original_file), "-ac", "1", str(temp_mono_file)]
        mono_result = subprocess.run(to_mono, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if mono_result.returncode != 0 or not temp_mono_file.exists():
            error_msg = mono_result.stderr.decode('utf-8', 'ignore')
            log(f"Failed converting to mono for file: {original_file}", indent=indent)
            raise ValueError(f"Error converting to mono: {error_msg}")

        audio_codec = "libmp3lame" if enhanced_audio_path.suffix.lower() == ".mp3" else "pcm_s16le"
        to_norm = ["ffmpeg-normalize", str(temp_mono_file), "-o", str(temp_norm_file), "-c:a", audio_codec, "-f"]
        try_with_range_target = subprocess.run(
            to_norm + ['--keep-loudness-range-target', '--sample-rate', str(df_state.sr())],
            capture_output=True
        )

        if try_with_range_target.returncode != 0:
            try_with_lra_20 = subprocess.run(
                to_norm + ['--loudness-range-target', '20', '--sample-rate', str(df_state.sr())],
                capture_output=True
            )

            if try_with_lra_20.returncode != 0 and not temp_norm_file.exists():
                error_msg = try_with_lra_20.stderr.decode('utf-8', 'ignore')
                log(f"ffmpeg-normalize failed completely: {error_msg}", indent=indent)
                raise ValueError(f"Error normalizing audio: {error_msg}")

        normalized_audio, _ = load_audio(temp_norm_file, sr=df_state.sr())
        enhanced_audio = enhance(model, df_state, normalized_audio)
        save_audio(enhanced_audio_path, enhanced_audio, sr=df_state.sr())

        if not enhanced_audio_path.exists():
            log(f"Failed writing enhanced audio for file: {original_file}", indent=indent)
            raise ValueError("Error writing enhanced audio")
    finally:
        temp_mono_file.unlink(missing_ok=True)
        temp_norm_file.unlink(missing_ok=True)


def generate_enhanced_audios(dataset_output_dir, files):
    log("ENHANCED Starting enhancement process for the whole dataset...", indent=2)
    enhanced_output_path = Path(dataset_output_dir, 'enhanced-audios')
    enhanced_output_path.mkdir(parents=True, exist_ok=True)
    model = None
    for file in tqdm.tqdm(files):
        original_file = Path(file)
        enhanced_file = enhanced_output_path / original_file.name
        if not enhanced_file.exists():
            if model is None:
                model, df_state = start_enhance_process()
            enhance_single_audio(original_file, enhanced_file, model=model, df_state=df_state, indent=2)


def generate_enhanced_metadata(experiment_path, base_subsets):
    enhanced_metadata_names = []
    for base_subset in base_subsets:
        subset_metadata = load_pickle(experiment_path / f'subset-{base_subset}' / 'metadata.pkl', cache=True)
        subset_metadata['file'] = subset_metadata['file'].apply(lambda path: resolve_audio_path(path, experiment_path / 'enhanced-audios'))
        subset_name = f'enhanced-{base_subset}'
        save_pickle(subset_metadata, experiment_path / f'subset-{subset_name}', 'metadata')
        enhanced_metadata_names.append(subset_name)
    return enhanced_metadata_names
