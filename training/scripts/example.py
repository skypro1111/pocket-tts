"""Example script showing how to prepare data and finetune on a small dataset."""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_dataset():
    """Create a dummy dataset for testing the training pipeline."""
    output_dir = Path("data/example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata
    metadata = [
        {
            "audio_path": "sample1.wav",
            "text": "Hello, this is a test of the text to speech system.",
            "language": "en",
        },
        {
            "audio_path": "sample2.wav",
            "text": "The quick brown fox jumps over the lazy dog.",
            "language": "en",
        },
        {
            "audio_path": "sample3.wav",
            "text": "Machine learning is transforming the world.",
            "language": "en",
        },
    ]
    
    # Save metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Created example dataset at {output_dir}")
    logger.info("Note: You need to add actual audio files to use this dataset")
    logger.info("Expected audio files:")
    for item in metadata:
        logger.info(f"  - {output_dir / item['audio_path']}")


def main():
    logger.info("Creating example dataset...")
    create_dummy_dataset()
    
    logger.info("\n" + "="*60)
    logger.info("Next steps:")
    logger.info("="*60)
    logger.info("\n1. Add audio files to data/example/")
    logger.info("   (sample1.wav, sample2.wav, sample3.wav)")
    
    logger.info("\n2. Validate the dataset:")
    logger.info("   python training/scripts/prepare_data.py validate --data_dir data/example")
    
    logger.info("\n3. Run finetuning:")
    logger.info("   python training/finetune.py \\")
    logger.info("     --data_dir data/example \\")
    logger.info("     --language en \\")
    logger.info("     --batch_size 1 \\")
    logger.info("     --num_epochs 5 \\")
    logger.info("     --output_dir outputs/example_model")
    
    logger.info("\n4. Evaluate the model:")
    logger.info("   python training/scripts/evaluate.py \\")
    logger.info("     --checkpoint outputs/example_model/final_model.pt \\")
    logger.info("     --test_data data/example/metadata.json \\")
    logger.info("     --output_dir outputs/example_evaluation")
    
    logger.info("\n" + "="*60)


if __name__ == "__main__":
    main()
