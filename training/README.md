# Finetuning Pocket TTS for Other Languages

This guide explains how to finetune Pocket TTS for languages other than English or adapt it to specific domains.

## Overview

Pocket TTS can be finetuned on new languages or domains using the provided training scripts. The model uses:
- **FlowLM**: A transformer-based flow matching model that generates audio latents
- **Mimi Codec**: A neural audio codec for encoding/decoding audio
- **SentencePiece Tokenizer**: For text tokenization

## Requirements

Install additional training dependencies:

```bash
pip install torchaudio tensorboard tqdm pyyaml
```

For distributed training:
```bash
pip install torch.distributed
```

## Quick Start

### 1. Prepare Your Dataset

Your dataset should be organized with audio files and a `metadata.json` file:

```
data/
├── french/
│   ├── audio1.wav
│   ├── audio2.wav
│   ├── ...
│   └── metadata.json
```

The `metadata.json` format:
```json
[
  {
    "audio_path": "audio1.wav",
    "text": "Bonjour, comment allez-vous?",
    "language": "fr"
  },
  {
    "audio_path": "audio2.wav",
    "text": "Je suis content de vous rencontrer.",
    "language": "fr"
  }
]
```

### 2. Using Data Preparation Scripts

#### Mozilla Common Voice

Prepare Common Voice dataset:

```bash
python training/scripts/prepare_data.py common-voice \
  --data_dir /path/to/cv-corpus-XX-YYYY \
  --output_dir data/french \
  --language fr \
  --split train
```

#### Custom Dataset

For custom audio and transcript files:

```bash
python training/scripts/prepare_data.py custom \
  --audio_dir /path/to/audio/files \
  --transcripts /path/to/transcripts.txt \
  --output_dir data/custom \
  --language fr
```

The transcripts file format (one per line):
```
audio1|Bonjour, comment allez-vous?
audio2|Je suis content de vous rencontrer.
```

#### Validate Dataset

Verify your dataset is properly formatted:

```bash
python training/scripts/prepare_data.py validate \
  --data_dir data/french
```

### 3. Simple Finetuning

For single-language finetuning without distributed training:

```bash
python training/finetune.py \
  --data_dir data/french \
  --language fr \
  --batch_size 2 \
  --num_epochs 10 \
  --lr 5e-5 \
  --output_dir outputs/french_model
```

This script:
- Uses a simplified training loop
- Works on single GPU or CPU
- Saves checkpoints every 500 steps
- Suitable for experimentation and small datasets

### 4. Advanced Training

For production training with distributed support:

```bash
# Single GPU
python training/train.py \
  --data_dir data/french \
  --languages fr \
  --batch_size 4 \
  --num_epochs 50 \
  --learning_rate 5e-5 \
  --output_dir outputs/french_model

# Multi-GPU (2 GPUs)
torchrun --nproc_per_node=2 training/train.py \
  --data_dir data/french \
  --languages fr \
  --batch_size 4 \
  --num_epochs 50 \
  --learning_rate 5e-5 \
  --world_size 2 \
  --output_dir outputs/french_model
```

### 5. Multilingual Training

Train on multiple languages simultaneously:

```bash
python training/train.py \
  --data_dir data/multilingual \
  --languages fr es de it \
  --batch_size 8 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --output_dir outputs/multilingual_model
```

## Configuration Files

Example configurations are provided in `training/configs/`:

- `french.yaml` - French language training
- `spanish.yaml` - Spanish language training
- `german.yaml` - German language training
- `multilingual.yaml` - Multi-language training

Use configuration files:

```bash
python training/train.py @training/configs/french.yaml
```

## Training Parameters

### Key Hyperparameters

- `--batch_size`: Batch size per GPU (default: 4)
  - Reduce if you run out of memory
  - Increase with `--accumulation_steps` for larger effective batch size

- `--learning_rate`: Learning rate (default: 5e-5)
  - Lower for finetuning (1e-5 to 5e-5)
  - Higher for training from scratch (1e-4 to 5e-4)

- `--num_epochs`: Number of training epochs (default: 50)
  - More epochs for smaller datasets
  - Fewer epochs for large datasets to avoid overfitting

- `--freeze_encoder`: Freeze Mimi encoder weights
  - Use when you only want to adapt the language model
  - Faster training and lower memory usage

- `--warmup_steps`: Learning rate warmup steps (default: 1000)
  - Helps with training stability

- `--gradient_clip`: Gradient clipping value (default: 1.0)
  - Prevents exploding gradients

### Distributed Training

- `--world_size`: Number of GPUs to use
- `--num_workers`: Number of data loading workers

## Dataset Recommendations

### Size Requirements

- **Minimum**: 1-2 hours of speech data
  - Suitable for finetuning on a new accent or style
  
- **Recommended**: 10-20 hours of speech data
  - Good quality multilingual adaptation
  
- **Optimal**: 50+ hours of speech data
  - High-quality language adaptation

### Quality Requirements

- **Sample Rate**: 16kHz or higher (will be resampled to 24kHz)
- **Audio Format**: WAV, MP3, or FLAC
- **Duration**: 2-15 seconds per sample (optimal)
- **Quality**: Clear speech with minimal background noise
- **Diversity**: Multiple speakers, various speaking styles

## Monitoring Training

Tensorboard logs are saved to `{output_dir}/logs`:

```bash
tensorboard --logdir outputs/french_model/logs
```

Monitor:
- Training loss
- Learning rate schedule
- Gradient norms

## Using Finetuned Models

After training, load your finetuned model:

```python
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.config import load_config
import torch

# Load config
config = load_config("pocket_tts/config/b6369a24.yaml")

# Create model
model = TTSModel._from_pydantic_config(
    config,
    temp=1.0,
    lsd_decode_steps=4,
    noise_clamp=None,
    eos_threshold=0.5
)

# Load finetuned weights
checkpoint = torch.load("outputs/french_model/final_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Use for inference
voice_state = model.get_state_for_audio_prompt("voice.wav")
audio = model.generate_audio(voice_state, "Bonjour tout le monde!")
```

## Supported Languages

The model architecture supports any language, but you need:

1. Training data in the target language
2. A SentencePiece tokenizer trained on the target language (or multilingual tokenizer)
3. Sufficient training data for good quality

### Pre-trained Language Support

Currently, Pocket TTS is pre-trained on:
- English (en)

### Easily Adaptable Languages

Languages with abundant TTS datasets:
- French (fr) - Common Voice, M-AILABS
- Spanish (es) - Common Voice, M-AILABS
- German (de) - Common Voice, M-AILABS
- Italian (it) - Common Voice, M-AILABS
- Portuguese (pt) - Common Voice
- Polish (pl) - Common Voice
- Dutch (nl) - Common Voice
- Russian (ru) - Common Voice
- Chinese (zh) - AISHELL, Common Voice
- Japanese (ja) - JVS, Common Voice
- Korean (ko) - KSS, Common Voice

## Troubleshooting

### Out of Memory

- Reduce `--batch_size`
- Enable `--freeze_encoder`
- Use gradient accumulation: `--accumulation_steps 2`

### Poor Quality Output

- Increase training data
- Train for more epochs
- Check data quality (audio clarity, transcription accuracy)
- Adjust `--learning_rate` (try lower values)

### Slow Training

- Increase `--num_workers` for faster data loading
- Use smaller `--max_audio_length` to reduce computation
- Enable `--freeze_encoder` to reduce trainable parameters

## Loss Functions

The training scripts implement the CALM (Continuous Audio Language Models) loss functions from the paper:

### Consistency Model Loss (CALMLoss)

The main training objective from Equation 3 of the paper:

```
LCALM(θ, ϕ, ψ) = Σs Et,ε [
    exp(wψ(t)) ||Fϕ(x^s_t, t, Z^s) - F̄ϕ(x^s_t, t, Z^s) - cos(t)∂f̄ϕ/∂t||²₂
    - wψ(t)
]
```

Features:
- **Adaptive Weighting**: Learnable wψ(t) function to balance loss at different timesteps
- **TrigFlow Schedule**: Uses αt = cos(t·π/2), σt = sin(t·π/2) for noise interpolation
- **EOS Prediction**: Binary cross-entropy loss for end-of-sequence detection

### Flow Matching Loss (Alternative)

A simpler alternative using the MAR-style diffusion objective:
```
L_diff(θ, ϕ) = Σs E_ε,t [||ε - εϕ(x^s_t, z^s, t)||²]
```

Use `--use_consistency=False` to switch to flow matching loss.

### Noise Augmentation

Following the paper, noise augmentation is applied to backbone input:
- Controlled by `--use_noise_augmentation` flag
- Maximum noise level set by `--max_noise_level` (default: 0.5)
- Encourages the backbone to focus on coarse structure

## Contributing

We welcome contributions to improve the training scripts! Areas for improvement:

- Add evaluation metrics (MOS, WER, etc.)
- Support for more data formats
- Better logging and visualization
- Hyperparameter tuning utilities
- Multi-GPU optimization

## References

- [Pocket TTS Paper](https://arxiv.org/abs/2509.06926)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Consistency Models](https://arxiv.org/abs/2303.01469)
