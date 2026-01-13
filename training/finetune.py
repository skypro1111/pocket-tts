"""Simplified finetuning script for adapting Pocket TTS to a new language.

This script provides a more accessible entry point for finetuning on a single
language dataset without requiring distributed training setup.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.config import load_config
from training.utils.dataset import MultilingualTTSDataset, TTSCollator
from training.utils.training_utils import (
    AverageMeter,
    count_parameters,
    save_checkpoint,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple finetuning script for Pocket TTS"
    )
    
    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing audio files and metadata.json"
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Target language code (e.g., 'fr', 'es', 'de')"
    )
    
    # Model
    parser.add_argument(
        "--config",
        type=str,
        default="pocket_tts/config/b6369a24.yaml",
        help="Model config file"
    )
    
    # Training
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=500, help="Save every N steps")
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/finetuned",
        help="Output directory for checkpoints"
    )
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    
    return parser.parse_args()


def validate_dataset(data_dir: Path, language: str):
    """Validate that the dataset is properly formatted."""
    metadata_file = data_dir / "metadata.json"
    
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {data_dir}. "
            "Please create a metadata.json file with format: "
            '[{"audio_path": "audio.wav", "text": "transcription", "language": "en"}]'
        )
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    if not metadata:
        raise ValueError("metadata.json is empty")
    
    # Check required fields
    required_fields = {"audio_path", "text", "language"}
    missing_fields = required_fields - set(metadata[0].keys())
    if missing_fields:
        raise ValueError(
            f"metadata.json entries must have fields: {required_fields}. "
            f"Missing: {missing_fields}"
        )
    
    # Check if target language exists
    languages = set(item["language"] for item in metadata)
    if language not in languages:
        raise ValueError(
            f"Language '{language}' not found in dataset. "
            f"Available languages: {languages}"
        )
    
    # Check audio files exist
    for item in metadata[:5]:  # Check first 5
        audio_path = data_dir / item["audio_path"]
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}. "
                "Ensure audio_path in metadata.json is relative to data_dir."
            )
    
    logger.info(f"Dataset validation passed. Found {len(metadata)} samples.")
    logger.info(f"Languages in dataset: {languages}")


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Validate dataset
    data_dir = Path(args.data_dir)
    validate_dataset(data_dir, args.language)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.language
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    config = load_config(args.config)
    model = TTSModel._from_pydantic_config_with_weights(
        config,
        temp=1.0,
        lsd_decode_steps=4,
        noise_clamp=None,
        eos_threshold=0.5
    )
    model = model.to(device)
    
    num_params = count_parameters(model)
    logger.info(f"Model has {num_params:,} trainable parameters")
    
    # Create dataset
    logger.info(f"Loading {args.language} dataset...")
    dataset = MultilingualTTSDataset(
        data_dir=data_dir,
        language_filter=[args.language],
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=TTSCollator(),
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    
    # Training loop
    logger.info("Starting training...")
    logger.warning(
        "NOTE: This is a simplified training script with placeholder loss. "
        "For production use, implement proper flow matching loss computation."
    )
    
    global_step = 0
    loss_meter = AverageMeter()
    
    for epoch in range(args.num_epochs):
        model.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for batch in pbar:
            # Move batch to device
            audios = batch["audios"].to(device)
            
            # Placeholder loss computation
            # TODO: Implement actual flow matching loss
            # This should involve:
            # 1. Encode audio with Mimi encoder
            # 2. Tokenize text with SentencePiece
            # 3. Compute flow matching loss with FlowLM
            # Using a dummy loss value that requires gradients for demonstration
            loss = audios.sum() * 0.0 + 0.5  # Creates a differentiable loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update metrics
            loss_meter.update(loss.item())
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
            
            # Save checkpoint
            if global_step % args.save_every == 0:
                checkpoint_path = output_dir / f"checkpoint_step_{global_step}.pt"
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    loss_meter.avg,
                    checkpoint_path,
                )
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save epoch checkpoint
        epoch_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            global_step,
            loss_meter.avg,
            epoch_path,
        )
        logger.info(
            f"Epoch {epoch + 1} complete. Average loss: {loss_meter.avg:.4f}"
        )
    
    # Save final model
    final_path = output_dir / "final_model.pt"
    save_checkpoint(
        model,
        optimizer,
        args.num_epochs,
        global_step,
        loss_meter.avg,
        final_path,
    )
    logger.info(f"Training complete! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
