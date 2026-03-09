import glob
from pathlib import Path
import pandas as pd

from src.dataset_readers.denoising import generate_enhanced_audios, generate_enhanced_metadata
from src.utils import log, save_pickle, load_samples_to_filter


def get_speech_segments(manual_aligns_basepath):
    log("MANUAL SEGMENTATION", indent=2)
    aligns = []
    for aligns_path in glob.glob(f'{manual_aligns_basepath}/**/*.csv', recursive=True):
        manual_aligns = pd.read_csv(aligns_path, header=0, index_col=0)
        manual_aligns['start'] = manual_aligns['begin'] / 1000  
        manual_aligns['end'] = manual_aligns['end'] / 1000     
        manual_aligns = manual_aligns.sort_values('start')
        
        speech_segments = manual_aligns[(manual_aligns['speaker'] == 'PAR') | (manual_aligns['speaker'] == 'INV')]
        if len(speech_segments) == 0 or 'PAR' not in speech_segments.speaker.unique():
            log(f"Warning: No subject speech segments found for sample {Path(aligns_path).stem} in {aligns_path}. "
                f"So, ignoring this sample for manual segmentation.", indent=2)
            continue
            
        speech_segments['sample_id'] = Path(aligns_path).stem
        speech_segments['duration'] = speech_segments['end'] - speech_segments['start']
        aligns.append(speech_segments[['sample_id', 'start', 'end', 'duration']])
    
    if len(aligns) == 0:
        raise ValueError(f"No valid manual alignments found in {manual_aligns_basepath}. Please check the base path.")
    log(f"✓ Loaded manual segmentation for {len(aligns)} samples from {manual_aligns_basepath}", indent=2)
    return pd.concat(aligns, ignore_index=True)


def read_audios_and_metadata(audio_path, samples_to_ignore, output_path):
    data = []
    AD = glob.glob(f'{audio_path}/ad/*.wav') + glob.glob(f'{audio_path}/ad/*.mp3')
    CTR = glob.glob(f'{audio_path}/cn/*.wav') + glob.glob(f'{audio_path}/cn/*.mp3')
    for file in AD + CTR:
        filename = Path(file).stem.replace('.wav', '')
        condition = 0 if file.split('/')[-2] == 'cn' else 1
        subject = filename
        if subject not in samples_to_ignore:
            data.append([file, filename, subject, condition])
        else:
            log(f"Warning: Subject {subject} is in the filter list. Ignoring this subject for dataset loading.")

    output_path.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(data, columns=['file', 'sample_id', 'subject', 'condition'])
    save_pickle(data, output_path, 'metadata')
    return data


def read_dataset(output_path, params):
    samples_to_ignore, _ = load_samples_to_filter(params.get('samples_to_filter'))
    log(f"✓ Filtered out {len(samples_to_ignore)} samples from metadata", indent=2)    

    list_to_return = []
    #### Original audios ################################################################################### 
    if params.get('original_audio_path'):
        data = read_audios_and_metadata(params['original_audio_path'], samples_to_ignore, output_path / 'subset-original')
        list_to_return.append('original')
    #### Challenge audios ##################################################################################
    if params.get('challenge_audio_path'):
        read_audios_and_metadata(params['challenge_audio_path'], samples_to_ignore, output_path / 'subset-challenge')
        list_to_return.append('challenge')
    
    #### Enhanced audios ################################################################################### 
    if params.get('original_audio_path') and params.get('apply_enhance'):        
        generate_enhanced_audios(output_path, data.file.unique())
        list_to_return.extend(generate_enhanced_metadata(output_path, ['original'])) 
    ########################################################################################################

    #### Load and save manual aligns for the whole dataset (will be used for all subsets) ##################
    if params.get('manual_aligns_path'):
        aligns = get_speech_segments(params['manual_aligns_path'])
        base_align_path = Path(output_path, 'manual-aligns')
        base_align_path.mkdir(parents=True, exist_ok=True)    
        save_pickle(aligns, base_align_path / 'manual_segmentation-only_speech.pkl')
    ########################################################################################################

    return list_to_return 
