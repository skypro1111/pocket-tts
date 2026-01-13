"""Data preparation utilities for creating training datasets."""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torchaudio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_common_voice_dataset(
    data_dir: str,
    output_dir: str,
    language: str,
    split: str = "train",
    max_samples: Optional[int] = None,
):
    """Prepare Mozilla Common Voice dataset for training.
    
    Common Voice dataset structure:
    - clips/ (audio files)
    - train.tsv, dev.tsv, test.tsv
    
    Args:
        data_dir: Path to Common Voice dataset directory
        output_dir: Output directory for prepared dataset
        language: Language code (e.g., 'en', 'fr', 'es')
        split: Which split to use ('train', 'dev', 'test')
        max_samples: Maximum number of samples (for testing)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read TSV file
    tsv_file = data_dir / f"{split}.tsv"
    if not tsv_file.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_file}")
    
    logger.info(f"Reading {tsv_file}...")
    
    metadata = []
    with open(tsv_file, "r", encoding="utf-8") as f:
        # Skip header
        header = f.readline().strip().split("\t")
        
        # Find column indices
        path_idx = header.index("path")
        sentence_idx = header.index("sentence")
        
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            parts = line.strip().split("\t")
            if len(parts) <= max(path_idx, sentence_idx):
                continue
            
            audio_filename = parts[path_idx]
            text = parts[sentence_idx]
            
            # Check if audio file exists
            audio_path = data_dir / "clips" / audio_filename
            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_path}")
                continue
            
            # Copy or link audio file
            output_audio_path = output_dir / audio_filename
            if not output_audio_path.exists():
                output_audio_path.parent.mkdir(parents=True, exist_ok=True)
                # Create symlink to save space
                try:
                    output_audio_path.symlink_to(audio_path.absolute())
                except (FileExistsError, OSError) as e:
                    # Symlink may already exist or filesystem doesn't support symlinks
                    logger.debug(f"Could not create symlink for {audio_filename}: {e}")
                    # Fall back to copying if symlink fails
                    import shutil
                    shutil.copy2(audio_path, output_audio_path)
            
            metadata.append({
                "audio_path": audio_filename,
                "text": text,
                "language": language,
            })
    
    # Save metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Prepared {len(metadata)} samples")
    logger.info(f"Metadata saved to {metadata_file}")


def prepare_custom_dataset(
    audio_dir: str,
    transcripts_file: str,
    output_dir: str,
    language: str,
    audio_extension: str = ".wav",
):
    """Prepare a custom dataset from audio files and transcripts.
    
    Expected format:
    - Audio files in audio_dir (e.g., 0001.wav, 0002.wav, ...)
    - Transcripts file with format: "filename|transcript" per line
    
    Args:
        audio_dir: Directory containing audio files
        transcripts_file: File with transcripts (format: filename|transcript)
        output_dir: Output directory for prepared dataset
        language: Language code
        audio_extension: Audio file extension
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Reading transcripts from {transcripts_file}...")
    
    metadata = []
    with open(transcripts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            
            parts = line.split("|", 1)
            if len(parts) != 2:
                logger.warning(f"Invalid line: {line}")
                continue
            
            filename, text = parts
            
            # Add extension if not present
            if not filename.endswith(audio_extension):
                filename = filename + audio_extension
            
            audio_path = audio_dir / filename
            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_path}")
                continue
            
            # Verify audio is loadable
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
            except Exception as e:
                logger.warning(f"Failed to load {audio_path}: {e}")
                continue
            
            # Copy or link audio file
            output_audio_path = output_dir / filename
            if not output_audio_path.exists():
                try:
                    output_audio_path.symlink_to(audio_path.absolute())
                except (FileExistsError, OSError) as e:
                    # Symlink may already exist or filesystem doesn't support symlinks
                    logger.debug(f"Could not create symlink for {filename}: {e}")
                    # Fall back to copying if symlink fails
                    import shutil
                    shutil.copy2(audio_path, output_audio_path)
            
            metadata.append({
                "audio_path": filename,
                "text": text,
                "language": language,
                "duration": waveform.shape[1] / sample_rate,
            })
    
    # Save metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Prepared {len(metadata)} samples")
    logger.info(f"Total duration: {sum(m['duration'] for m in metadata) / 3600:.2f} hours")
    logger.info(f"Metadata saved to {metadata_file}")


def validate_dataset(data_dir: str):
    """Validate a prepared dataset."""
    data_dir = Path(data_dir)
    
    metadata_file = data_dir / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata.json not found in {data_dir}")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    logger.info(f"Dataset contains {len(metadata)} samples")
    
    # Check languages
    languages = set(item["language"] for item in metadata)
    logger.info(f"Languages: {languages}")
    
    # Check audio files
    missing_files = []
    for item in metadata:
        audio_path = data_dir / item["audio_path"]
        if not audio_path.exists():
            missing_files.append(audio_path)
    
    if missing_files:
        logger.warning(f"Missing {len(missing_files)} audio files:")
        for path in missing_files[:10]:
            logger.warning(f"  {path}")
    else:
        logger.info("All audio files present")
    
    # Print statistics
    logger.info(f"\nStatistics:")
    for lang in languages:
        lang_samples = [m for m in metadata if m["language"] == lang]
        logger.info(f"  {lang}: {len(lang_samples)} samples")
        
        if "duration" in lang_samples[0]:
            total_duration = sum(m["duration"] for m in lang_samples)
            logger.info(f"    Duration: {total_duration / 3600:.2f} hours")
            logger.info(
                f"    Average: {total_duration / len(lang_samples):.2f} seconds"
            )


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for training")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Common Voice
    cv_parser = subparsers.add_parser("common-voice", help="Prepare Common Voice dataset")
    cv_parser.add_argument("--data_dir", required=True, help="Common Voice data directory")
    cv_parser.add_argument("--output_dir", required=True, help="Output directory")
    cv_parser.add_argument("--language", required=True, help="Language code")
    cv_parser.add_argument("--split", default="train", help="Split to use")
    cv_parser.add_argument("--max_samples", type=int, help="Maximum samples")
    
    # Custom dataset
    custom_parser = subparsers.add_parser("custom", help="Prepare custom dataset")
    custom_parser.add_argument("--audio_dir", required=True, help="Audio directory")
    custom_parser.add_argument("--transcripts", required=True, help="Transcripts file")
    custom_parser.add_argument("--output_dir", required=True, help="Output directory")
    custom_parser.add_argument("--language", required=True, help="Language code")
    custom_parser.add_argument("--extension", default=".wav", help="Audio extension")
    
    # Validate
    val_parser = subparsers.add_parser("validate", help="Validate dataset")
    val_parser.add_argument("--data_dir", required=True, help="Dataset directory")
    
    args = parser.parse_args()
    
    if args.command == "common-voice":
        prepare_common_voice_dataset(
            args.data_dir,
            args.output_dir,
            args.language,
            args.split,
            args.max_samples,
        )
    elif args.command == "custom":
        prepare_custom_dataset(
            args.audio_dir,
            args.transcripts,
            args.output_dir,
            args.language,
            args.extension,
        )
    elif args.command == "validate":
        validate_dataset(args.data_dir)


if __name__ == "__main__":
    main()
