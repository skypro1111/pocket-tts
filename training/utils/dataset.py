"""Dataset utilities for multilingual TTS training."""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MultilingualTTSDataset(Dataset):
    """Dataset for multilingual TTS training.
    
    Expected data format:
    - metadata.json with entries: {"audio_path": "...", "text": "...", "language": "..."}
    - audio files in various formats (wav, mp3, flac)
    
    Args:
        data_dir: Directory containing audio files and metadata
        metadata_file: Name of the metadata JSON file
        sample_rate: Target sample rate for audio
        max_audio_length: Maximum audio length in seconds (for memory efficiency)
        language_filter: Optional list of languages to include
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        metadata_file: str = "metadata.json",
        sample_rate: int = 24000,
        max_audio_length: float = 30.0,
        language_filter: Optional[list[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.language_filter = language_filter
        
        # Load metadata
        metadata_path = self.data_dir / metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        # Filter by language if specified
        if language_filter:
            self.metadata = [
                item for item in self.metadata 
                if item.get("language") in language_filter
            ]
        
        logger.info(f"Loaded {len(self.metadata)} samples from {data_dir}")
        
        # Build language to ID mapping
        languages = sorted(set(item["language"] for item in self.metadata))
        self.language_to_id = {lang: idx for idx, lang in enumerate(languages)}
        self.id_to_language = {idx: lang for lang, idx in self.language_to_id.items()}
        
        logger.info(f"Languages: {languages}")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single training sample.
        
        Returns:
            dict with keys: audio, text, language, language_id, audio_path
        """
        item = self.metadata[idx]
        
        # Load audio
        audio_path = self.data_dir / item["audio_path"]
        waveform, orig_sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if orig_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sample_rate,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        # Truncate if too long
        max_samples = int(self.max_audio_length * self.sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        
        # Squeeze to 1D
        waveform = waveform.squeeze(0)
        
        return {
            "audio": waveform,
            "text": item["text"],
            "language": item["language"],
            "language_id": self.language_to_id[item["language"]],
            "audio_path": str(audio_path),
        }


class TTSCollator:
    """Collator for batching TTS samples with padding."""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: list[dict]) -> dict:
        """Collate a batch of samples.
        
        Args:
            batch: List of samples from MultilingualTTSDataset
            
        Returns:
            Batched tensors with padding
        """
        # Extract fields
        audios = [item["audio"] for item in batch]
        texts = [item["text"] for item in batch]
        languages = [item["language"] for item in batch]
        language_ids = torch.tensor([item["language_id"] for item in batch])
        
        # Pad audios to same length
        audio_lengths = torch.tensor([audio.shape[0] for audio in audios])
        max_audio_len = audio_lengths.max().item()
        
        padded_audios = torch.zeros(len(audios), max_audio_len)
        for i, audio in enumerate(audios):
            padded_audios[i, :audio.shape[0]] = audio
        
        return {
            "audios": padded_audios,
            "audio_lengths": audio_lengths,
            "texts": texts,
            "languages": languages,
            "language_ids": language_ids,
        }
