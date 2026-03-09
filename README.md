# Spurious Correlation Detection Toolkit

This repository introduces a toolkit for uncovering spurious correlations between recording conditions and target class in speech datasets.

## Overview

Spurious correlations are common in speech corpora, especially in health-related datasets, when recording conditions vary with target labels (e.g., room noise, device/channel effects, encoding artifacts, or capture protocol differences).
When these correlations are present in both train and test data, system performance can be overestimated, especially in high-stakes contexts.

The toolkit performs a diagnostic test by predicting the target class using only the **non-speech** regions of each recording. Better-than-chance performance indicates that target-relevant information leaks through non-speech artifacts.

# Spurious Correlation Detection Toolkit

This repository introduces a toolkit for uncovering spurious correlations between recording conditions and target class in speech datasets.

## Overview

Spurious correlations are common in speech corpora, especially in health-related datasets, when recording conditions vary with target labels (e.g., room noise, device/channel effects, encoding artifacts, or capture protocol differences).
When these correlations are present in both train and test data, system performance can be overestimated, especially in high-stakes contexts.

The toolkit performs a diagnostic test by predicting the target class using only the **non-speech** regions of each recording. Better-than-chance performance indicates that target-relevant information leaks through non-speech artifacts.

## Installation

Clone the repository and install the Python dependencies:

```bash
git clone URL
cd spurious-correlation-toolkit

pip install -r requirements.txt
```

## How to run

Single experiment:

```bash
python src/pipeline.py -c -cf configs/SpanishAD/speech-mfcc.yaml
```

Run all non-template configs:

```bash
bash run.sh
```

`-c` enables cache mode to reuse existing artifacts.

## What this repo includes

- A reusable, dataset-agnostic experiment pipeline.
- Dataset readers for supported datasets.
- VAD/alignment modules and manual VAD verification tools.
- Feature extraction modules.
- Training and evaluation logic with bootstrapping.
- Configuration templates to run experiments quickly.
- Analysis notebooks.

## Repository structure

- `src/pipeline.py`: main script.
- `src/dataset_readers/`: dataset-specific loaders.
  - `adresso.py`, `spanishad.py`
- `src/speech_alignments/`: VAD/alignment methods.
  - `silero.py`, `whisper.py`, `pyannote.py`, `speechbrain.py`, `torchvad.py`
  - plus utilities and post-processing.
- `src/features/`: feature extractors.
  - `mfcc.py`, `spectrogram.py`, `melspectrogram.py`, `wav2vec.py`.
- `src/model_development/`: split creation, datasets, models, training, metrics, bootstrap.
- `configs/`: experiment configuration files.
  - `base-mfcc.yaml`
  - `base-spectrogram.yaml`
  - `base-wav2vec.yaml`
  - dataset folders `configs/ADReSSo/`, `configs/SpanishAD/`
- `run.sh`: script to run all available configs.
- `notebooks/`: analysis helpers and result plotting.
  - `01-VAD-Manual-Review.ipynb`
  - `02-Metadata-Leakage-Audit.ipynb`
  - `03-Results-Summary.ipynb`
- `requirements.txt`: Python dependencies.

## Method

1. **Non-speech extraction**: get non-speech regions from VAD or manual annotations.
2. **Feature extraction**: extract acoustic features from each defined segment. 
3. **Concatenation + chunking**: build fixed-length chunks so the model does not directly use global duration/timing.
4. **Training/inference**: train a classifier on chunk-level samples; average chunk scores per waveform.
5. **Decision**: if non-speech classification is above chance, likely indicates recording-condition leakage.

## Important assumptions

The method is designed for datasets where:

- recordings contain some non-speech/silence time,
- labels are available at waveform level,
- task can be framed as binary classification (or one-vs-rest for multiclass).

## Typical use cases

Useful as a preprocessing sanity check before final modeling to:

- detect protocol/recording artifacts linked to labels,
- decide if additional data collection controls are needed,
- compare preprocessing variants (raw/challenge/enhanced, different VADs).

## VAD / non-speech quality control

Reliable non-speech boundaries are central to valid diagnosis.

- Supported VADs in this repo:
  - Pyannote, Silero, Whisper, TorchVAD, SpeechBrain
- The toolkit provides:
  - VAD quality review against manually annotated examples,
  - speech leakage and missed non-speech metrics,
  - manual listening/annotation tool to audit non-speech segments.

Recommended control:

- use manual checks when possible,
- when needed, run a second VAD stage on first-pass non-speech,
- discard samples with clear speech leakage from subsequent analysis.

## Pipeline (configurable components)

This section maps the YAML sections to the runtime pipeline.

### Config structure (base)

The base starting templates are:
- `configs/base-mfcc.yaml`
- `configs/base-spectrogram.yaml`
- `configs/base-wav2vec.yaml`

### Global controls

Set experiment-level options and output location.

```yaml
experiment_output_dir: IS26-experiments/
```

Choose which alignment-derived segments are used:

```yaml
only_alignment_contains: non_speech
```

or speech-only:

```yaml
exclude_alignment_contains: non_speech
```

Use one of these options (or leave both unset to use both speech and non-speech samples).

### Dataset reader

Select the dataset reader and dataset paths/flags.

```yaml
dataset:
  name: dataset_name
  metadata: path/to/metadata.pkl
  apply_enhance: true
```

Dataset readers are defined in `src/dataset_readers/` and can be extended for new corpora.
Each reader should return available `subset`s and generate the per-subset `metadata.pkl` files used downstream.

### Speech enhancement

Configured in the dataset configuration (`apply_enhance`), the enhancement step uses:

- loudness normalization,
- DeepFilterNet,

implemented in `src/dataset_readers/denoising.py`.

### Resampling

Audio is first downsampled to the minimum required sample rate in preprocessing, then upsampled to `16000` Hz for downstream normalization/feature consistency.

### VAD / manual alignment / full-audio mode

Use one or more aligners in order:

```yaml
aligners:
  - name: manual
    params:
      filename: manual_ipu-only_patient
```

`full` is also supported in configs; in that mode, each whole audio file is processed from start to end and then split by the configured `segmenter` parameters.
For any non-full mode, non-speech regions are computed as the gaps between speech regions and then also split by `segmenter`.

### Split strategy

```yaml
splits:
  folds_amount: 8
  repetitions: 10
  group_column: subject
```

Set `folds_amount: 1` to use a single train/test split.

### Feature extraction

The toolkit uses local-frame features and chunk-based training.

```yaml
features:
  - name: mfcc
    params:
      n_mfcc: 40
      sample_rate: 16000
      melkwargs:
        n_fft: 400
        hop_length: 160
        n_mels: 80
        center: false
        power: 2.0
        mel_scale: slaney
        norm: slaney
      concatenate_segments: false
      segmenter:
        size: 5
        overlap: 4
```

Multiple feature entries can be declared in one config, but one model definition is used for all listed features.

### Modeling

Use one model block per experiment.

```yaml
model:
  name: AcousticCNNClassifier
  n_jobs: 2
  group_column: subject
  shuffle: true
  wrap_last: true
  batch_size: 32
  epochs: 80
  dropout: 0.3
  projection_dim: 128
  early_stopping:
    patience: 10
    metric: loss
  bootstrapping:
    n_bootstraps: 1000
    stratify: true
```

## Metrics

- Main metric: AUC (ROC AUC). But we also calculate accuracy. 
- Confidence intervals: bootstrap over test predictions.
- The metric is used as a sanity signal, not as a final absolute benchmark by itself.

## VAD Decision

The `notebooks/01-VAD-Manual-Review.ipynb` notebook is the diagnostic step for VAD quality:

- compare automatic VAD outputs against manual annotations;
- quantify speech leakage / non-speech recall;

Use this notebook to make an explicit VAD decision before running full classification experiments.

## Audio Metadata

Use `notebooks/02-Metadata-Leakage-Audit.ipynb` to audit audio-level metadata and detect technical spurious correlations:

- load all `metadata.pkl` files from an experiment root,
- extract per-file properties with `ffmpeg.probe` (codec, sample rate, channels, bit depth, etc.),
- compare metadata distributions across condition/class and experiment subsets,
- run a simple RandomForest baseline over metadata features (100 train/test repetitions) with Accuracy and AUC,
- visualize distribution balance and metadata-driven classification results to identify recording confounds.
