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
from pocket_tts.modules.stateful_module import init_states
from pocket_tts.utils.config import load_config
from training.utils.dataset import MultilingualTTSDataset, TTSCollator
from training.utils.losses import CALMLoss
from training.utils.noise_schedule import BackboneNoiseAugmentation, EMAStatistics
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

    # Loss function
    parser.add_argument(
        "--use_consistency",
        action="store_true",
        default=True,
        help="Use consistency loss (default). If False, uses flow matching."
    )
    parser.add_argument(
        "--use_noise_augmentation",
        action="store_true",
        default=True,
        help="Apply noise augmentation to backbone input"
    )

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


def compute_batch_loss(model, batch, device, loss_fn, noise_augmentation, ema_stats):
    """Compute training loss for a batch using CALM methodology.

    Args:
        model: The TTSModel.
        batch: Batch dict with 'audios', 'texts', 'audio_lengths'.
        device: Device to run on.
        loss_fn: The CALMLoss instance.
        noise_augmentation: Optional BackboneNoiseAugmentation.
        ema_stats: Optional EMAStatistics for normalization.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    audios = batch["audios"].to(device)
    texts = batch["texts"]
    audio_lengths = batch["audio_lengths"].to(device)

    # Step 1: Encode audio to latents
    with torch.no_grad():
        if audios.dim() == 2:
            audios = audios.unsqueeze(1)  # [B, 1, T]
        latents = model.mimi.encode_to_latent(audios)  # [B, D, S]
        latents = latents.transpose(-1, -2)  # [B, S, D]

    # Compute sequence lengths in latent space
    frame_rate = model.config.mimi.frame_rate
    sample_rate = model.config.mimi.sample_rate
    latent_lengths = (audio_lengths.float() / sample_rate * frame_rate).long()

    # Step 2: Normalize latents
    if ema_stats is not None:
        ema_stats.update(latents)
        latents_normalized = ema_stats.normalize(latents)
    else:
        latents_normalized = (latents - model.flow_lm.emb_mean) / model.flow_lm.emb_std

    # Step 3: Apply noise augmentation
    if noise_augmentation is not None:
        backbone_input, _ = noise_augmentation(latents_normalized)
    else:
        backbone_input = latents_normalized

    # Step 4: Get text embeddings
    batch_size, seq_len, latent_dim = latents_normalized.shape
    text_embeddings_list = []
    for text in texts:
        prepared = model.flow_lm.conditioner.prepare(text)
        text_emb = model.flow_lm.conditioner(prepared)
        text_embeddings_list.append(text_emb)

    max_text_len = max(emb.shape[1] for emb in text_embeddings_list)
    text_embeddings = torch.zeros(
        batch_size, max_text_len, model.flow_lm.dim,
        device=device, dtype=latents.dtype
    )
    for i, emb in enumerate(text_embeddings_list):
        text_embeddings[i, :emb.shape[1]] = emb[0]

    # Step 5: Create shifted input for autoregressive modeling
    if seq_len > 1:
        bos = torch.full(
            (batch_size, 1, latent_dim), float("nan"),
            device=device, dtype=backbone_input.dtype
        )
        backbone_input_shifted = torch.cat([bos, backbone_input[:, :-1]], dim=1)
    else:
        backbone_input_shifted = torch.full(
            (batch_size, 1, latent_dim), float("nan"),
            device=device, dtype=backbone_input.dtype
        )

    # Run through backbone
    input_transformed = model.flow_lm.input_linear(backbone_input_shifted)
    combined_input = torch.cat([text_embeddings, input_transformed], dim=1)

    model_state = init_states(model.flow_lm, batch_size=batch_size, sequence_length=1000)
    transformer_out = model.flow_lm.transformer(combined_input, model_state)
    if model.flow_lm.out_norm:
        transformer_out = model.flow_lm.out_norm(transformer_out)

    conditioning = transformer_out[:, text_embeddings.shape[1]:]

    # Step 6: Compute EOS targets
    eos_targets = torch.zeros(batch_size, seq_len, device=device)
    for i, length in enumerate(latent_lengths):
        if length > 0 and length <= seq_len:
            eos_targets[i, length - 1] = 1.0

    eos_logits = model.flow_lm.out_eos(conditioning).squeeze(-1)

    # Step 7: Compute loss
    loss, metrics = loss_fn(
        flow_net=model.flow_lm.flow_net,
        conditioning=conditioning,
        x_clean=latents_normalized,
        eos_logits=eos_logits,
        eos_targets=eos_targets,
    )

    return loss, metrics


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

    # Create loss function
    logger.info(f"Creating loss function (consistency={args.use_consistency})")
    loss_fn = CALMLoss(
        use_consistency=args.use_consistency,
        eos_weight=0.1,
        adaptive_weight=True,
    )
    loss_fn = loss_fn.to(device)

    # Create noise augmentation if enabled
    noise_augmentation = None
    if args.use_noise_augmentation:
        logger.info("Using noise augmentation")
        noise_augmentation = BackboneNoiseAugmentation(
            min_noise_level=0.0,
            max_noise_level=0.5,
        )

    # Create EMA statistics tracker
    latent_dim = config.mimi.quantizer.dimension
    ema_stats = EMAStatistics(dim=latent_dim, device=device)

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

    # Create optimizer - include loss function parameters
    params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        betas=(0.9, 0.95),  # Paper uses β1=0.9, β2=0.95
        weight_decay=0.01,
    )

    # Training loop
    logger.info("Starting training with CALM loss...")

    global_step = 0
    loss_meter = AverageMeter()
    metrics_meters = {}

    for epoch in range(args.num_epochs):
        model.train()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for batch in pbar:
            # Compute loss using CALM methodology
            loss, metrics = compute_batch_loss(
                model=model,
                batch=batch,
                device=device,
                loss_fn=loss_fn,
                noise_augmentation=noise_augmentation,
                ema_stats=ema_stats,
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update metrics
            loss_meter.update(loss.item())
            for key, value in metrics.items():
                if key not in metrics_meters:
                    metrics_meters[key] = AverageMeter()
                metrics_meters[key].update(value)
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
