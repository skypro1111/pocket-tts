# Quick Start Guide: Finetuning Pocket TTS

This guide will help you quickly get started with finetuning Pocket TTS on your own data.

## Installation

1. Install Pocket TTS with training dependencies:

```bash
pip install pocket-tts
pip install -r training/requirements.txt
```

Or using uv:

```bash
uv pip install pocket-tts
uv pip install -r training/requirements.txt
```

## 5-Minute Example

### Step 1: Create Example Dataset

```bash
python training/scripts/example.py
```

This creates a template dataset structure in `data/example/`.

### Step 2: Add Your Audio Files

Add audio files (WAV, MP3, or FLAC) matching the metadata:
- `data/example/sample1.wav`
- `data/example/sample2.wav`
- `data/example/sample3.wav`

### Step 3: Validate Dataset

```bash
python training/scripts/prepare_data.py validate --data_dir data/example
```

### Step 4: Start Finetuning

```bash
python training/finetune.py \
  --data_dir data/example \
  --language en \
  --batch_size 1 \
  --num_epochs 5 \
  --output_dir outputs/my_model
```

### Step 5: Test Your Model

```python
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.config import load_config
import torch

# Load model
config = load_config("pocket_tts/config/b6369a24.yaml")
model = TTSModel._from_pydantic_config(
    config, temp=1.0, lsd_decode_steps=4,
    noise_clamp=None, eos_threshold=0.5
)

# Load finetuned weights
checkpoint = torch.load("outputs/my_model/final_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Generate speech
voice_state = model.get_state_for_audio_prompt("voice.wav")
audio = model.generate_audio(voice_state, "Hello from my finetuned model!")
```

## Real Dataset Example: French

### 1. Download Common Voice

Download French dataset from [Mozilla Common Voice](https://commonvoice.mozilla.org/):

```bash
# Download and extract cv-corpus-XX-YYYY-XX-fr.tar.gz
tar -xzf cv-corpus-XX-YYYY-XX-fr.tar.gz
```

### 2. Prepare Dataset

```bash
python training/scripts/prepare_data.py common-voice \
  --data_dir cv-corpus-XX-YYYY-XX/fr \
  --output_dir data/french \
  --language fr \
  --split train \
  --max_samples 10000
```

### 3. Finetune on French

```bash
python training/finetune.py \
  --data_dir data/french \
  --language fr \
  --batch_size 4 \
  --num_epochs 20 \
  --lr 5e-5 \
  --output_dir outputs/french_model
```

### 4. Monitor Training

```bash
tensorboard --logdir outputs/french_model/logs
```

### 5. Evaluate

```bash
python training/scripts/evaluate.py \
  --checkpoint outputs/french_model/final_model.pt \
  --test_data data/french/metadata.json \
  --output_dir outputs/french_evaluation \
  --max_samples 10
```

## Custom Dataset Example

### 1. Organize Your Data

```
my_dataset/
├── audio/
│   ├── file001.wav
│   ├── file002.wav
│   └── ...
└── transcripts.txt
```

`transcripts.txt` format:
```
file001|This is the first sentence.
file002|This is the second sentence.
```

### 2. Prepare Dataset

```bash
python training/scripts/prepare_data.py custom \
  --audio_dir my_dataset/audio \
  --transcripts my_dataset/transcripts.txt \
  --output_dir data/my_dataset \
  --language en
```

### 3. Finetune

```bash
python training/finetune.py \
  --data_dir data/my_dataset \
  --language en \
  --batch_size 2 \
  --num_epochs 15 \
  --output_dir outputs/my_custom_model
```

## Multi-Language Example

### 1. Prepare Multiple Languages

```bash
# French
python training/scripts/prepare_data.py common-voice \
  --data_dir cv-corpus/fr --output_dir data/multi/fr --language fr

# Spanish  
python training/scripts/prepare_data.py common-voice \
  --data_dir cv-corpus/es --output_dir data/multi/es --language es

# German
python training/scripts/prepare_data.py common-voice \
  --data_dir cv-corpus/de --output_dir data/multi/de --language de
```

### 2. Merge Datasets

```bash
# Create combined directory
mkdir -p data/multilingual

# Merge metadata files
python -c "
import json
from pathlib import Path

metadata = []
for lang in ['fr', 'es', 'de']:
    with open(f'data/multi/{lang}/metadata.json') as f:
        metadata.extend(json.load(f))

with open('data/multilingual/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"

# Link audio files
for lang in fr es de; do
  ln -s $(pwd)/data/multi/$lang/*.wav data/multilingual/
done
```

### 3. Train Multilingual Model

```bash
python training/train.py \
  --data_dir data/multilingual \
  --languages fr es de \
  --batch_size 8 \
  --num_epochs 50 \
  --learning_rate 1e-4 \
  --output_dir outputs/multilingual_model
```

## Tips for Success

### Data Quality
- Use clean audio with minimal background noise
- Ensure accurate transcriptions
- Aim for 2-15 seconds per audio sample
- Include diverse speakers and speaking styles

### Training
- Start with a small dataset to test the pipeline
- Use lower learning rates (1e-5 to 5e-5) for finetuning
- Monitor training loss in TensorBoard
- Save checkpoints frequently

### Troubleshooting
- **Out of memory**: Reduce batch_size
- **Slow training**: Increase num_workers
- **Poor quality**: Train longer or add more data
- **Model diverges**: Lower learning rate

## Next Steps

- Read the full [training README](README.md) for detailed information
- Check example configurations in `training/configs/`
- Experiment with different hyperparameters
- Evaluate on held-out test data

## Support

For issues or questions:
- Check the [main README](../README.md)
- Open an issue on [GitHub](https://github.com/kyutai-labs/pocket-tts/issues)
- Read the [paper](https://arxiv.org/abs/2509.06926) for technical details
