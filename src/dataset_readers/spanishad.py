import glob
from pathlib import Path
import pandas as pd

from src.utils import log, save_pickle, load_samples_to_filter
from src.dataset_readers.denoising import generate_denoised_audios, generate_denoised_metadata


def balance_dataset(dataset_params, metadata):
    original_metadata = pd.read_csv(dataset_params['metadata'])
    original_metadata['subject'] = original_metadata.subject.str.replace('_', '-')
    original_metadata = original_metadata.drop_duplicates(subset='subject').set_index('subject')
    metadata['gender'] = metadata.subject.apply(lambda s: original_metadata.loc[s]['gender'])
    
    min_counts = metadata.groupby(['gender', 'condition']).size().groupby(level=0).min()    
    subjects = []
    for gender, count in min_counts.items():
        gender_df = metadata[metadata.gender == gender].copy()
        balanced_gender = gender_df.groupby('condition').apply(lambda x: x.sample(n=int(count), random_state=42), include_groups=False).subject
        subjects.extend(balanced_gender.tolist())
    return metadata[metadata.subject.isin(subjects)].reset_index(drop=True)


def read_dataset(output_path, params):
    samples_to_ignore, _ = load_samples_to_filter(params.get('samples_to_filter'))
    log(f"✓ Filtered out {len(samples_to_ignore)} samples from metadata", indent=2)    
    
    metadata = []
    AD = glob.glob(f'{params["base_audio_path"]}/AD_*_lamina_1.wav')
    CTR = glob.glob(f'{params["base_audio_path"]}/CTR_*_lamina_1.wav')
    for file in AD + CTR:
        sample_id = Path(file).stem
        if sample_id in samples_to_ignore:
            log(f"Warning: Sample {sample_id} is in the filter list. Ignoring this sample for dataset loading.")
            continue
        condition = 0 if sample_id.split('_')[0] == 'CTR' else 1
        subject = '-'.join(sample_id.split('_')[0:2])
        metadata.append({'file': file, 'sample_id': sample_id, 'subject': subject, 'condition': condition})
    
    metadata = pd.DataFrame(metadata)
    metadata = balance_dataset(params, metadata)
    (output_path / 'subset-original').mkdir(parents=True, exist_ok=True)
    save_pickle(metadata, output_path / 'subset-original', 'metadata')
    
    subsets = ['original']
    #### Denoised audios ################################################################################### 
    if params.get('apply_denoise'):        
        generate_denoised_audios(output_path, metadata.file.unique())
        subsets.extend(generate_denoised_metadata(output_path, subsets))        
    ########################################################################################################
         
    ########### Load and save manual aligns for the whole dataset (will be used for all subsets) ###########
    if params.get('manual_aligns'):
        aligns = pd.read_pickle(params['manual_aligns'])
        aligns = aligns[aligns.filename.str.contains('lamina_1')]
        aligns['sample_id'] = aligns['filename'].str.replace('-', '_')
        aligns = aligns[['sample_id', 'start', 'end', 'content', 'interviewer']]
        
        base_align_path = Path(output_path, 'manual-aligns')
        base_align_path.mkdir(parents=True, exist_ok=True)    
        aligns[(aligns.content != 'SIL') | (aligns.interviewer)].to_pickle(base_align_path / 'manual_ipu.pkl')
        aligns[(aligns.content != 'SIL') & (~aligns.interviewer)].to_pickle(base_align_path / 'manual_ipu-only_patient.pkl')
    ########################################################################################################
    
    return subsets