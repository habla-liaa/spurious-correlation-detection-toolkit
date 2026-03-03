import tqdm
import pandas as pd

from pathlib import Path
from pydub import AudioSegment
from pydub.playback import play
from argparse import ArgumentParser

import yaml

from src.utils import log
from src.pipeline import load_dataset
from src.speech_alignments.vad_engine import load_or_create_alignments


def concatenate_audio_segments(audio_file, segments):
    audio = AudioSegment.from_file(audio_file)
            
    concatenated = AudioSegment.empty()
    for _, segment in segments.sort_values('start').iterrows():
        start_ms = int(segment['start'] * 1000)
        end_ms = int(segment['end'] * 1000)
        concatenated += audio[start_ms:end_ms]

    return concatenated


def get_user_comment(sample_id):
    log("-" * 60, newline=True)
    log(f"Sample ID: {sample_id}")
    log("-" * 60)
    log("Options:")
    log("  - Press ENTER to skip (no comment)")
    log("  - Type your comment and press ENTER")
    log("  - Type 'replay' to hear the audio again")
    log("  - Type 'quit' to finish and save")

    comment = input("\nYour comment: ").strip()
    return comment


def export_alignments(alignment_dir, aligns, output_path):
    total_samples = len(aligns['file'].unique())
    alignment_name = str(alignment_dir).replace("/", "_")
    export_dir = output_path / 'exported_alignments' / alignment_name
    export_dir.mkdir(parents=True, exist_ok=True)
        
    for (sample_id, file), segments in tqdm.tqdm(aligns.groupby(['sample_id', 'file']), total=total_samples):
        concatenated_audio = concatenate_audio_segments(file, segments)
        
        output_file = export_dir / f"{sample_id}.wav"
        
        concatenated_audio.export(output_file, format="wav")
    
    log("=" * 60, newline=True)
    log("Export completed!")
    log(f"  {export_dir}")
    
    return True


def review_alignments(alignment_dir, aligns, output_path):
    total_samples = len(aligns['file'].unique())
    comments = []

    if (output_path / 'review_comments.csv').exists():
        comments = pd.read_csv(output_path / 'review_comments.csv')
        aligns = aligns[~aligns['sample_id'].isin(comments['sample_id'].unique())]
        log(f"Resuming review. {len(comments['sample_id'].unique())} sample(s) already reviewed.")

    for idx, ((sample_id, file), segments) in enumerate(aligns.groupby(['sample_id', 'file'])):
        log(f"[{idx}/{total_samples}] Processing: {sample_id}", newline=True)
        
        log(f"Concatenating {len(segments)} segment(s)...")
        concatenated_audio = concatenate_audio_segments(file, segments)        
        
        while True:
            play(concatenated_audio)
            comment = get_user_comment(sample_id)
            
            if comment.lower() == 'quit':
                log("Quitting review...", newline=True)
                return False
            elif comment.lower() == 'replay':
                continue
            else:
                if isinstance(comments, list):
                    comments = pd.DataFrame(comments)
                comments = pd.concat([comments, pd.DataFrame([{
                    'sample_id': sample_id,
                    'audio_file': file,
                    'aligner': str(alignment_dir),
                    'comment': comment,
                }])], ignore_index=True)
                comments.to_csv(output_path / 'review_comments.csv', index=False)    
                break
    
    log("=" * 60, newline=True)
    log(f"Review completed! Reviewed {len(comments)} audio file(s)")
    log("=" * 60)
    log("", newline=True)

    return True


def display_alignments(alignments):
    log("=" * 60, newline=True)
    log("AVAILABLE ALIGNMENTS")
    log("=" * 60)
    
    for idx, align_dir in enumerate(alignments, 1):
        log(f"{idx}. {align_dir}")
    log("=" * 60)
    log("Type 'quit' to exit the reviewer.")


def find_or_generate_alignments(config):
    all_aligns_in_config = []
    
    config = {k: config[k] for k in config if k in ['experiment_output_dir', 'dataset', 'aligners']}
    dataset_output_dir, subset_names = load_dataset(config)
    
    for subset_name in subset_names:
        log(f"SUBSET: {subset_name}", indent=2)
        subset_output_dir = dataset_output_dir / f'subset-{subset_name}'
        for aligner_cfg in config["aligners"]:
            aligns_dirs = load_or_create_alignments(subset_output_dir, aligner_cfg['name'], aligner_cfg.get('params', {}), indent=4)            
            aligns_dirs = [align_dir for align_dir in aligns_dirs if 'non_speech' in align_dir.name]
            all_aligns_in_config.extend(aligns_dirs)
    
    return all_aligns_in_config


def select_mode():
    log("\n" + "="*60)
    log("SELECT MODE")
    log("="*60)
    log("1. Review Mode - Play audio with comments")
    log("2. Export Mode - Generate and save audio files")
    log("="*60)
    
    while True:
        choice = input("\nSelect mode (1 or 2): ").strip()
        if choice == '1':
            return 'review'
        elif choice == '2':
            return 'export'
        else:
            log("Invalid choice! Please enter 1 or 2.")


def main(config):

    log("=" * 60)
    log("AUDIO ALIGNMENT REVIEWER")
    log("=" * 60)
    
    mode = select_mode()
    
    log("\nSearching for alignments...")
    alignments = find_or_generate_alignments(config)
    
    output_path = Path(config['experiment_output_dir'], 'alignment_reviews')
    output_path.mkdir(parents=True, exist_ok=True)        
    while True:
        display_alignments(alignments)
        log("Select an alignment (number):", newline=True)
        choice = input().strip()

        if choice.lower() == 'quit':
            log("Exiting reviewer...")
            break
        
        try:
            align_idx = int(choice) - 1
        except ValueError:
            log("Invalid choice! Please enter a number.")
            continue
            
        if align_idx < 0 or align_idx >= len(alignments):
            log("Invalid choice!")
            continue
        
        current_alignment_dir = alignments[align_idx]

        log(f"Loading alignment: {current_alignment_dir}...", newline=True)
        aligns_df = pd.read_pickle(current_alignment_dir / 'aligns.pkl')
        
        log(f"Loaded {len(aligns_df['file'].unique())} files with alignments")

        if mode == 'review':
            completed = review_alignments(current_alignment_dir, aligns_df, output_path)
            if not completed:
                break
        else:
            export_alignments(current_alignment_dir, aligns_df, output_path)
    
        finish_review = input("Do you want to process another alignment? (y/n): ").strip().lower()
        if finish_review != 'y':
            break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-cf', '--config', dest='config', required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    
    if config.get('aligners', []) == []:
        log("No aligners specified in config! Please add aligners to the configuration file.")
    else:
        main(config)
