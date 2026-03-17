# Spurious Correlation Detection Toolkit

This repository introduces a toolkit for uncovering spurious correlations between recording conditions and target class in speech datasets.

## Index

- [Overview](#overview)
- [Installation](#installation)
- [How to run](#how-to-run)
- [What this repo includes](#what-this-repo-includes)
- [Repository structure](#repository-structure)
- [Method](#method)
- [Important assumptions](#important-assumptions)
- [Use cases](#use-cases)
- [VAD / non-speech quality control](#vad--non-speech-quality-control)
- [Pipeline](#pipeline)
  - [Config structure](#config-structure)
  - [Global controls](#global-controls)
  - [Dataset reader](#dataset-reader)
  - [VAD / manual alignment / full-audio mode](#vad--manual-alignment--full-audio-mode)
  - [Split strategy](#split-strategy)
  - [Feature extraction](#feature-extraction)
  - [Modeling](#modeling)
  - [Metrics](#metrics)
- [Notebooks](#notebooks)
  - [VAD Decision](#vad-decision)
  - [Audio Metadata](#audio-metadata)
  - [Results](#results)

## Overview

Spurious correlations appear in speech corpora when recording conditions (e.g., room noise, device/channel effects, encoding artifacts, or capture protocol differences) correlate with target labels. This may happen, for example, in health-care collections, if control subjects are collected in a different location or by a different doctor with a different device than patients with the target condition. 

When these correlations are present in both train and test data, system performance can be overestimated, especially in high-stakes contexts.

The toolkit performs a diagnostic test by predicting the target class using only the **non-speech** regions of each recording. Better-than-chance performance indicates that target-relevant information leaks through non-speech artifacts.

## Installation

Clone the repository and install the Python dependencies:

```bash
git clone URL
cd spurious-correlation-detection-toolkit

pip install -r requirements.txt
```

## How to run

Single experiment:

```bash
python src/pipeline.py --cache --config configs/SpanishAD/speech-mfcc.yaml
```

Run all configs in the config directory:

```bash
bash run.sh
```

`--cache` enables cache mode to reuse existing files from prior runs.

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
  - template configs
  - dataset folders `configs/ADReSSo/`, `configs/SpanishAD/`
- `run.sh`: script to run all available configs.
- `notebooks/`: analysis helpers and result plotting.
  - `01-VAD-Manual-Review.ipynb`
  - `02-Metadata-Leakage-Audit.ipynb`
  - `03-Results-Summary.ipynb`
- `requirements.txt`: Python dependencies.

## Method

The method consists of the following steps:

1. **Non-speech extraction**: get non-speech regions from VAD or manual annotations.
2. **Feature extraction**: extract acoustic features from each defined segment. 
3. **Concatenation + chunking**: build fixed-length chunks so the model does not directly use global duration/timing.
4. **Training/inference**: train a classifier on chunk-level samples; during inference, average chunk scores per waveform.
5. **Decision**: if non-speech classification is above chance, likely indicates the presence of spurious correlations between the recording conditions and the sample class.

## Important assumptions

The method is designed for datasets where:

- recordings contain at least a few seconds of non-speech/silence, 
- the class of interest is annotated at waveform level,
- task can be framed as binary classification (or one-vs-rest for multiclass).

## Use cases

This toolkit can be used as a preprocessing sanity check before using datasets for speech processing tasks, specially when the recording conditions were not fully controlled. It allows users to detect protocol/recording artifacts correlated to labels. If used during data collection, a positive result from this sanity check would signal that the collection protocol needs improvement.

## VAD / non-speech quality control

Reliable non-speech boundaries are central to valid diagnosis.

To support that validation process, the repository includes several automatic VAD backends (Pyannote, Silero, Whisper, TorchVAD, and SpeechBrain), together with utilities that let you compare automatic VADs against manual annotations, quantify speech leakage and missed non-speech regions, and run listening-based audits so that decisions about which aligner to use.

We recommend users to

- Manually annotate some samples for VAD when possible, to allow for tuning of VAD system parameters. The speech leakage should be very small to allow for valid results when using this tool.
- When needed, run a second VAD stage on first-pass non-speech.
- Manually check the resulting VAD and discard samples with speech leakage from subsequent analysis.

## Pipeline

This section describes the sections in the YAML configs. This config allow to connect all parts of the pipeline. 

### Config structure

When starting with new experiments, first you have to define a config file. It is divided in:

- [Global controls](#global-controls): defines the experiment-level settings.
- [Dataset reader](#dataset-reader): prepares metadata and audio paths for the experiments. In this part, you define or extend the dataset reader for the corpus you want to use, and you can optionally enable a speech-enhanced version of the audio.
- [VAD / manual alignment / full-audio mode](#vad--manual-alignment--full-audio-mode): defines which alignment strategy is used. You can optionally include clipping functionality.
- [Split strategy](#split-strategy): defines how aligned samples are split for training, validation, and test.
- [Feature extraction](#feature-extraction): defines one or multiple feature extractors, applied to each previously selected aligner output.
- [Modeling](#modeling): defines the model architecture and training configuration for the pipeline.
- [Metrics](#metrics): define how model performance is summarized and how optional bootstrapping is configured.


### Global controls

Set experiment-level options and output location.

```yaml
experiment_output_dir: IS26-experiments/
```

Choose which alignment-derived segments are used, if non-speech or speech-only:

```yaml
keep_segments_that_contain: [non_speech|speech]
```
Leave it unset to use both speech and non-speech samples. If the pipeline will use the full audio, this condition will be ignore.

### Dataset reader

This section selects the dataset loader and passes the dataset-specific paths and options required to build the experiment metadata.

```yaml
dataset:
  name: dataset_name
  <dataset-specific parameters>
  apply_enhance: true
```

Dataset readers are implemented in `src/dataset_readers/`. Each reader must define a `read_dataset(output_path, params)` function, save a `metadata.pkl` file inside each generated subset directory, and return the list of subset names that will be used by the rest of the pipeline.

At minimum, the saved metadata must contain these columns:

- `file`
- `sample_id`
- `subject`
- `condition`

A minimal reader looks like this:

```python
from src.dataset_readers.enhancer import generate_enhanced_audios, generate_enhanced_metadata
from pathlib import Path
from src.utils import save_pickle

def read_dataset(output_path, params):
  metadata = upload_audios_and_metadata(params['original_audio_path'], output_path / 'subset-original')
  assert {'file', 'sample_id', 'subject', 'condition'}.issubset(metadata.columns)  
  list_to_return = ['original']
  save_pickle(metadata, output_path / 'subset-original', 'metadata')

  if params.get('apply_enhance'):
    generate_enhanced_audios(output_path, metadata.file.unique())
    list_to_return.extend(generate_enhanced_metadata(output_path, list_to_return))

  if params.get('manual_aligns_path'):
    aligns = prepare_manual_alignments(params['manual_aligns_path'])
    base_align_path = Path(output_path, 'manual-aligns')
    base_align_path.mkdir(parents=True, exist_ok=True)
    save_pickle(aligns, base_align_path / 'manual_segmentation.pkl')

  return list_to_return
```

In practice, the reader is the right place to organize file paths, filter samples, prepare manual alignments, and define alternative subsets such as original, challenge, or enhanced versions of the same corpus.


#### Speech enhancement

If `apply_enhance: true` is set, the pipeline generates enhanced audio versions with loudness normalization followed by DeepFilterNet, as implemented in `src/dataset_readers/enhancer.py`.


#### Resampling

Before alignment and feature extraction, each subset is resampled in `src/dataset_readers/preprocessing.py`: if multiple sample rates are found, audio is first adjusted to the minimum rate present in the subset and then converted to `16000` Hz. This step is applied automatically to all generated subsets.


### VAD / manual alignment / full-audio mode

Alignment can be configured in three complementary modes, and each mode answers a different experimental question, think this section as the part of the pipeline where you define exactly what temporal regions of every waveform are going to be exposed to the classifier.

The general structure for defining one or multiple aligners is:
```yaml
aligners:
  - name: <name of the system>
    params:
      <corresponding configuration>
  - name: <name of another system>
    params:
      <corresponding configuration>
  ...
```

If your configuration contains only aligners and does not include feature extraction or model settings, the pipeline will finish after generating and saving the alignment files.

You can also enable an optional clipping step to normalize segment duration, either using a target mean duration or a fixed number of seconds.


#### Full-audio
When you use the `full` aligner, no extra parameters are required, and each recording is represented by a single interval that starts at time `0` and ends at the total duration of the audio file.

```yaml
- name: full
```

#### Manual
When you use the `manual` aligner, you provide the filename of a previously prepared alignment table, and that file must be saved under the experiment output directory, typically as part of dataset preparation.

```yaml
aligners:
  - name: manual
    params:
      filename: manual_segmentation
```

#### Automatic Voice-Activity-Detector
For automatic VAD, the aligner name depends on the backend you choose.

- [Pyannote](https://github.com/pyannote/pyannote-audio), which requires a Hugging Face token.

```yaml
- name: pyannote
  params:
    huggingface_token: ''
```

- [Silero](https://github.com/snakers4/silero-vad), where you can set the speech decision threshold.

```yaml
- name: silero
  params:
    threshold: 0.2
```

- [SpeechBrain](https://github.com/speechbrain/speechbrain), with no additional required parameters.

- [TorchVAD](https://docs.pytorch.org/audio/main/generated/torchaudio.functional.vad.html), where `torchVAD_params` are passed directly to the underlying torchaudio function.

```yaml
- name: torchvad
  params:
    torchVAD_params:
      trigger_level: 5.5
      trigger_time: 0.1
      search_time: 0.5
      allowed_gap: 0.1
      measure_freq: 10.0
```
- [Whisper](https://github.com/openai/whisper). In this case, the system uses automatic transcription as an intermediate step, so you need to define the model to be used (together with the corresponding language), and the resulting transcription is then filtered so that only valid transcribed regions are retained as speech segments.

```yaml
- name: whisper
  params:
    model: large
    compute_type: float32
    language: es
    beam_size: 5
    no_speech_prob_threshold: 0.8
    compress_ratio_threshold: 2.4
```

For all automatic VAD detectors, you can define an `audio_path` so alignment is estimated from an alternative audio source (for example, enhanced audio) while features are still extracted from another source (for example, original audio), and this setup works as long as durations are consistent and filenames can be matched even if extensions differ.

All automatic systems first produce speech intervals, and then the pipeline computes non-speech intervals as the complement over each recording timeline, which guarantees that speech and non-speech partitions are coherent by construction; in addition, non-speech segments shorter than `0.5` seconds are removed to reduce unstable micro-segments.

Another option is post-processing, where a second VAD model is run only over first-pass non-speech intervals to catch residual speech leakage, and this allows combinations such as SpeechBrain in the first stage and Silero in the second stage. In practice, this is useful when the first detector is intentionally permissive and you want a second pass that is more conservative before accepting the final non-speech segments.

```yaml
- name: speechbrain
  params:
    postprocess:
      name: silero
      params:
        threshold: 0.2
```

If you want to add a new aligner backend, you can create a new module under `src/speech_alignments/` and reference that module filename as the aligner name in the configuration file.

### Split strategy
Splits are computed separately for each alignment output, so only samples that actually contain the corresponding aligned segments are considered in that experiment.

The split procedure uses stratified k-fold partitioning over the values defined by `group_column`, which is typically the speaker identifier, so train, validation, and test sets remain speaker-independent while preserving the class distribution given by `condition`.

You must define the number of folds, the number of repetitions, and the grouping column used to prevent leakage across partitions.

```yaml
splits:
  folds_amount: 8
  repetitions: 10
  group_column: subject
```

### Feature extraction

Feature extractors are implemented in `src/features/`. New extractors can be added as new modules and referenced by filename in the config.

The general structure is:

```yaml
features:
  - name: <feature extractor name>
    params:
      <all corresponding parameters>
  - name: <feature extractor name 2>
    params:
      <all corresponding parameters>
  ...
```

Features are extracted from the intervals produced by the selected aligner. The pipeline reads the waveform, cuts each aligned segment, and computes the representation only on that segment, which avoids using acoustic context outside the selected interval.

At the feature stage, you can also define segment concatenation and fixed-size chunking.

- [Melspectrogram](https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.Spectrogram.html)
This extractor uses `torchaudio.transforms.Spectrogram`, and its parameters are passed directly from the config.

```yaml
- name: spectrogram
  params:
    n_fft: 400
    hop_length: 160
    power: 2.0
```

- [MFCC](https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.MFCC.html)
This extractor uses `torchaudio.transforms.MFCC`, and the MFCC and Mel parameters are defined in the config.

```yaml
- name: mfcc
  params:
    n_mfcc: 40
    melkwargs:
      n_fft: 400
      hop_length: 160
      n_mels: 80
      center: false
      power: 2.0
      mel_scale: slaney
      norm: slaney
```

- [Wav2Vec 2.0](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
For Wav2Vec 2.0, you must define the Hugging Face model to use. The pipeline loads all hidden layers, concatenates them across the layer dimension, and then averages over time, so the final representation keeps layer information but not frame-level time resolution. When `concatenate_segments` or `segmenter` is used, those operations are applied first, and the temporal averaging is performed afterwards on the resulting sequence or chunk.

```yaml
- name: wav2vec
  params:
    model: ""
    minFrames: 512
```

#### Concatenation
If `concatenate_segments` is enabled, the feature representations extracted from all segments of the same `sample_id` are concatenated in temporal order before the model stage. This is mainly useful when the aligner returns multiple intervals per recording.

```yaml
  - name: wav2vec
    params:
      model: jonatasgrosman/wav2vec2-large-xlsr-53-spanish
      minFrames: 512
      concatenate_segments: true
```

#### Chunking
After feature extraction, and optionally after concatenation, you can apply a `segmenter` that divides each feature sequence into fixed-size windows with overlap. Each window is treated as an independent sample during training and evaluation.

```yaml
- name: wav2vec
  params:
    model: jonatasgrosman/wav2vec2-large-xlsr-53-spanish
    minFrames: 512
    concatenate_segments: true
    segmenter:
      size: 5
      overlap: 4
``` 

### Modeling

At the moment, each config defines a single model architecture. Because of that it is clearer to create one config per alignment condition and per feature type.

The repository includes two model architectures.

The model and training parameters are:

- `n_jobs`: number of folds processed in parallel for each repetition.
- `batch_size`: number of samples used in each training batch.
- `epochs`: maximum number of training epochs.
- `dropout`: dropout probability used in the projection layers.
- `shuffle`: if `true`, training samples are shuffled before batching.
- `wrap_last`: if `true`, the last incomplete training batch is completed by reusing samples from the beginning of the dataset.
- `drop_last`: if `true`, the last incomplete training batch is discarded.
- `device`: selects the execution device; if set to GPU, the pipeline uses CUDA when available.
- `learning_rate`: learning rate used by the Adam optimizer.

The options `wrap_last` and `drop_last` are mutually exclusive, and they only affect the training split.


#### Wav2Vec2Classifier:

`Wav2Vec2Classifier` is designed for Wav2Vec-based embeddings, where the feature dimension depends on the number of hidden layers used by the selected model.

```yaml
  name: Wav2Vec2Classifier
  n_jobs: 2
  shuffle: true
  wrap_last: true
  batch_size: 32
  epochs: 80
  dropout: 0.3
  num_layers: 25
  projection_dim: 64
```

The value of `num_layers` must match the Wav2Vec model used during feature extraction. The parameter `projection_dim` controls the hidden size of the projection layer before the final classifier, and `without_hidden: true` can be used to remove that projection stage and classify directly from the pooled embedding.

#### AcousticCNNClassifier:

`AcousticCNNClassifier` is intended for MFCC and spectrogram features, where each sample still preserves a temporal dimension.

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
```

For this architecture, `projection_dim` controls the size of the dense representation after temporal pooling, and `channel_multiplier` can be used to change the number of convolutional channels with respect to the input feature dimension.

#### Early stopping

Early stopping can be enabled inside the model configuration. In that case, the model is first trained using train/validation folds to select the best epoch according to validation `loss` or validation `auc`, and then it is retrained on the combined train+validation data using that selected number of epochs.

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
```

Available early-stopping parameters are `metric`, `min_epochs`, and `patience`.

- `metric`: validation criterion used to decide when training stops; it can be `loss` or `auc`.
- `min_epochs`: minimum number of epochs to run before early stopping is allowed.
- `patience`: number of non-improving epochs tolerated before stopping.

If early stopping is not enabled, the model is trained for the full number of epochs defined in `epochs`.

### Metrics
The pipeline reports metrics on the test split, with one result per repetition (`seed_*`). For each repetition, predictions from all folds are concatenated and then aggregated at `group_column` level by averaging probabilities inside each group, so evaluation remains speaker-level when `group_column` is the speaker identifier.

The reported metrics are:

- `auc`: ROC AUC computed from grouped probabilities.
- `acc`: grouped classification accuracy.
- `random_acc`: majority-class baseline accuracy for the same grouped labels.

Optional bootstrapping can be enabled inside the model section to estimate metric variability.

```yaml
model:
  name: AcousticCNNClassifier
  
  bootstrapping:
    n_bootstraps: 1000
    stratify: true
```

Bootstrapping is computed per repetition using grouped predictions. Set `stratify: true` to preserve class balance during bootstrap resampling.

## Notebooks

### VAD Decision

The `notebooks/01-VAD-Manual-Review.ipynb` notebook is the diagnostic step for VAD quality:

- compare automatic VAD outputs against manual annotations;
- quantify speech leakage / non-speech recall;

Use this notebook to make an explicit VAD decision before running full classification experiments.

In addition to notebook-based diagnostics, you can concatenate all non-speech intervals from each waveform into a single continuous audio file, which makes leakage patterns easier to detect during listening.

For that workflow, run:

```bash
python src/speech_alignments/vad_analysis.py --config configs/templates/aligners.yaml
```

The script can generate alignments when they are not already available, and then guides you through an interactive review mode (listen and annotate) or an export mode (save concatenated non-speech files for external listening), so you can combine quantitative and qualitative evidence before selecting a VAD setup.

### Audio Metadata

Use `notebooks/02-Metadata-Leakage-Audit.ipynb` to audit audio-level metadata and detect technical spurious correlations:

- load all `metadata.pkl` files from an experiment root,
- extract per-file properties with `ffmpeg.probe` (codec, sample rate, channels, bit depth, etc.),
- compare metadata distributions across condition and experiment subsets,
- run a simple RandomForest baseline over metadata features (100 train/test repetitions) with Accuracy and AUC,
- visualize distribution balance and metadata-driven classification results to identify recording confounds.

### Results

Use `notebooks/03-Results-Summary.ipynb` to load and summarize results for all experiments under the experiment output path.

