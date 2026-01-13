"""Main training script for finetuning Pocket TTS on multilingual data."""

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.config import load_config
from training.utils.dataset import MultilingualTTSDataset, TTSCollator
from training.utils.losses import CALMLoss
from training.utils.noise_schedule import BackboneNoiseAugmentation, EMAStatistics
from training.utils.training_utils import (
    AverageMeter,
    count_parameters,
    format_time,
    get_lr,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Pocket TTS on multilingual data")
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing training data"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="metadata.json",
        help="Name of metadata file"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help="List of languages to include (default: all)"
    )
    
    # Model arguments
    parser.add_argument(
        "--config",
        type=str,
        default="pocket_tts/config/b6369a24.yaml",
        help="Path to model config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze Mimi encoder during training"
    )
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation")
    
    # Logging and saving
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/training",
        help="Output directory"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log every N steps"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="Evaluate every N steps"
    )
    
    # Distributed training
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--world_size", type=int, default=1, help="Number of GPUs")

    # Loss function arguments
    parser.add_argument(
        "--use_consistency",
        action="store_true",
        default=True,
        help="Use consistency loss (default: True). If False, uses flow matching loss."
    )
    parser.add_argument(
        "--eos_weight",
        type=float,
        default=0.1,
        help="Weight for EOS prediction loss"
    )
    parser.add_argument(
        "--use_noise_augmentation",
        action="store_true",
        default=True,
        help="Apply noise augmentation to backbone input"
    )
    parser.add_argument(
        "--max_noise_level",
        type=float,
        default=0.5,
        help="Maximum noise level for backbone augmentation"
    )

    return parser.parse_args()


def compute_loss(
    model,
    batch,
    device,
    loss_fn,
    noise_augmentation=None,
    ema_stats=None,
):
    """Compute training loss for a batch using CALM methodology.

    This implements the full training pipeline from the CALM paper:
    1. Encode audio with Mimi encoder to get latent codes
    2. Normalize latents using EMA statistics
    3. Apply noise augmentation to backbone input
    4. Pass through backbone transformer to get conditioning
    5. Compute consistency/flow matching loss

    Args:
        model: The TTSModel containing flow_lm and mimi.
        batch: Batch dict with 'audios', 'texts', 'audio_lengths'.
        device: Device to run on.
        loss_fn: The CALMLoss instance.
        noise_augmentation: Optional BackboneNoiseAugmentation.
        ema_stats: Optional EMAStatistics for latent normalization.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    audios = batch["audios"].to(device)
    texts = batch["texts"]
    audio_lengths = batch["audio_lengths"].to(device)

    # Get actual model (unwrap DDP if needed)
    actual_model = model.module if isinstance(model, DDP) else model

    # Step 1: Encode audio to latents using Mimi encoder
    # Shape: [B, T_audio] -> [B, D, S] -> [B, S, D]
    with torch.no_grad():
        # Add channel dimension if needed
        if audios.dim() == 2:
            audios = audios.unsqueeze(1)  # [B, 1, T]

        # Encode with Mimi - this gives us the continuous latent representation
        latents = actual_model.mimi.encode_to_latent(audios)  # [B, D, S]
        latents = latents.transpose(-1, -2)  # [B, S, D]

    # Compute sequence lengths in latent space
    frame_rate = actual_model.config.mimi.frame_rate
    sample_rate = actual_model.config.mimi.sample_rate
    latent_lengths = (audio_lengths.float() / sample_rate * frame_rate).long()

    # Step 2: Normalize latents
    if ema_stats is not None:
        ema_stats.update(latents)
        latents_normalized = ema_stats.normalize(latents)
    else:
        # Use model's stored statistics
        latents_normalized = (latents - actual_model.flow_lm.emb_mean) / actual_model.flow_lm.emb_std

    # Step 3: Apply noise augmentation to backbone input (optional)
    if noise_augmentation is not None:
        backbone_input, noise_levels = noise_augmentation(latents_normalized)
    else:
        backbone_input = latents_normalized
        noise_levels = None

    # Step 4: Tokenize text and get text embeddings
    batch_size, seq_len, latent_dim = latents_normalized.shape

    # Prepare text conditioning
    # Tokenize each text in the batch
    text_embeddings_list = []
    for text in texts:
        prepared = actual_model.flow_lm.conditioner.prepare(text)
        text_emb = actual_model.flow_lm.conditioner(prepared)
        text_embeddings_list.append(text_emb)

    # Pad text embeddings to same length
    max_text_len = max(emb.shape[1] for emb in text_embeddings_list)
    text_embeddings = torch.zeros(
        batch_size, max_text_len, actual_model.flow_lm.dim,
        device=device, dtype=latents.dtype
    )
    for i, emb in enumerate(text_embeddings_list):
        text_embeddings[i, :emb.shape[1]] = emb[0]  # emb is [1, T, D]

    # Step 5: Run backbone transformer to get conditioning
    # The backbone processes: [text_embeddings, backbone_input] -> conditioning
    # We need to shift the sequence for autoregressive prediction

    # Create shifted input for autoregressive modeling
    # x[1:S] predicts x[2:S+1], so we use x[0:S-1] as input to predict x[1:S]
    if seq_len > 1:
        # Prepend BOS token (NaN signals BOS in the model)
        bos = torch.full(
            (batch_size, 1, latent_dim),
            float("nan"),
            device=device,
            dtype=backbone_input.dtype
        )
        backbone_input_shifted = torch.cat([bos, backbone_input[:, :-1]], dim=1)
        target_latents = latents_normalized  # Target is the full sequence
    else:
        backbone_input_shifted = torch.full(
            (batch_size, 1, latent_dim),
            float("nan"),
            device=device,
            dtype=backbone_input.dtype
        )
        target_latents = latents_normalized

    # Pass through input linear layer
    input_transformed = actual_model.flow_lm.input_linear(backbone_input_shifted)

    # Concatenate text embeddings with transformed input
    combined_input = torch.cat([text_embeddings, input_transformed], dim=1)

    # Run through transformer backbone (without using model state for training)
    # We create a dummy state for the forward pass
    from pocket_tts.modules.stateful_module import init_states
    model_state = init_states(actual_model.flow_lm, batch_size=batch_size, sequence_length=1000)

    transformer_out = actual_model.flow_lm.transformer(combined_input, model_state)
    if actual_model.flow_lm.out_norm:
        transformer_out = actual_model.flow_lm.out_norm(transformer_out)

    # Extract the conditioning corresponding to audio positions
    # (remove the text embedding positions)
    conditioning = transformer_out[:, text_embeddings.shape[1]:]

    # Step 6: Compute EOS targets
    # EOS is 1 at the last valid position, 0 elsewhere
    eos_targets = torch.zeros(batch_size, seq_len, device=device)
    for i, length in enumerate(latent_lengths):
        if length > 0 and length <= seq_len:
            eos_targets[i, length - 1] = 1.0

    # Get EOS predictions from the model
    eos_logits = actual_model.flow_lm.out_eos(conditioning).squeeze(-1)

    # Step 7: Compute loss
    loss, metrics = loss_fn(
        flow_net=actual_model.flow_lm.flow_net,
        conditioning=conditioning,
        x_clean=target_latents,
        eos_logits=eos_logits,
        eos_targets=eos_targets,
    )

    return loss, metrics


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    epoch,
    args,
    device,
    loss_fn,
    noise_augmentation=None,
    ema_stats=None,
    writer=None,
    global_step=0,
):
    """Train for one epoch using CALM loss."""
    model.train()

    loss_meter = AverageMeter()
    metrics_meters = {}
    epoch_start = time.time()

    for batch_idx, batch in enumerate(train_loader):
        # Compute loss using CALM methodology
        loss, metrics = compute_loss(
            model=model,
            batch=batch,
            device=device,
            loss_fn=loss_fn,
            noise_augmentation=noise_augmentation,
            ema_stats=ema_stats,
        )

        # Scale loss for gradient accumulation
        loss = loss / args.accumulation_steps
        loss.backward()

        # Update weights
        if (batch_idx + 1) % args.accumulation_steps == 0:
            # Gradient clipping
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.gradient_clip
                )

            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            global_step += 1

        # Update metrics
        loss_meter.update(loss.item() * args.accumulation_steps)

        # Update individual metric meters
        for key, value in metrics.items():
            if key not in metrics_meters:
                metrics_meters[key] = AverageMeter()
            metrics_meters[key].update(value)

        # Logging
        if (batch_idx + 1) % args.log_interval == 0:
            lr = get_lr(optimizer)
            elapsed = time.time() - epoch_start

            # Build log message with metrics
            metrics_str = " ".join(
                f"{k}: {v.avg:.4f}" for k, v in metrics_meters.items()
            )

            logger.info(
                f"Epoch [{epoch}] Step [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss_meter.avg:.4f} LR: {lr:.6f} "
                f"{metrics_str} Time: {format_time(elapsed)}"
            )

            if writer is not None:
                writer.add_scalar("train/loss", loss_meter.avg, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                # Log individual metrics
                for key, meter in metrics_meters.items():
                    writer.add_scalar(f"train/{key}", meter.avg, global_step)

        # Save checkpoint
        if global_step % args.save_interval == 0:
            save_path = Path(args.output_dir) / f"checkpoint_step_{global_step}.pt"
            save_checkpoint(
                model.module if isinstance(model, DDP) else model,
                optimizer,
                epoch,
                global_step,
                loss_meter.avg,
                save_path,
                scheduler,
            )

    return global_step


def main_worker(gpu, args):
    """Main training worker for distributed training."""
    # Setup distributed training
    if args.world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=gpu
        )
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed
    set_seed(args.seed + gpu)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup tensorboard
    writer = None
    if gpu == 0:
        writer = SummaryWriter(log_dir=output_dir / "logs")

    # Load config
    config = load_config(args.config)

    # Create model
    logger.info("Creating model...")
    model = TTSModel._from_pydantic_config_with_weights(
        config,
        temp=1.0,
        lsd_decode_steps=4,
        noise_clamp=None,
        eos_threshold=0.5
    )

    # Optionally freeze encoder
    if args.freeze_encoder:
        logger.info("Freezing Mimi encoder")
        for param in model.flow_lm.parameters():
            param.requires_grad = False

    model = model.to(device)

    # Wrap with DDP if distributed
    if args.world_size > 1:
        model = DDP(model, device_ids=[gpu])

    logger.info(f"Model has {count_parameters(model):,} trainable parameters")

    # Create loss function
    logger.info(f"Creating loss function (consistency={args.use_consistency})")
    loss_fn = CALMLoss(
        use_consistency=args.use_consistency,
        eos_weight=args.eos_weight,
        adaptive_weight=True,
    )
    loss_fn = loss_fn.to(device)

    # Create noise augmentation if enabled
    noise_augmentation = None
    if args.use_noise_augmentation:
        logger.info(f"Using noise augmentation with max_level={args.max_noise_level}")
        noise_augmentation = BackboneNoiseAugmentation(
            min_noise_level=0.0,
            max_noise_level=args.max_noise_level,
        )

    # Create EMA statistics tracker
    latent_dim = config.mimi.quantizer.dimension
    ema_stats = EMAStatistics(dim=latent_dim, device=device)

    # Create dataset
    logger.info("Loading dataset...")
    train_dataset = MultilingualTTSDataset(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        language_filter=args.languages,
    )

    # Create sampler
    sampler = None
    if args.world_size > 1:
        sampler = DistributedSampler(train_dataset, shuffle=True)

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=TTSCollator(),
        pin_memory=True,
    )

    # Create optimizer - include loss function parameters (adaptive weight)
    params_to_optimize = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),  # Paper uses β1=0.9, β2=0.95
    )

    # Create scheduler
    total_steps = len(train_loader) * args.num_epochs // args.accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.learning_rate * 0.1,
    )

    # Load checkpoint if provided
    start_epoch = 0
    global_step = 0
    if args.checkpoint:
        checkpoint = load_checkpoint(
            Path(args.checkpoint),
            model.module if isinstance(model, DDP) else model,
            optimizer,
            scheduler,
        )
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]

    # Training loop
    logger.info("Starting training with CALM loss...")
    for epoch in range(start_epoch, args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        global_step = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            args=args,
            device=device,
            loss_fn=loss_fn,
            noise_augmentation=noise_augmentation,
            ema_stats=ema_stats,
            writer=writer,
            global_step=global_step,
        )

        # Save epoch checkpoint
        if gpu == 0:
            save_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(
                model.module if isinstance(model, DDP) else model,
                optimizer,
                epoch,
                global_step,
                0.0,
                save_path,
                scheduler,
            )
    
    logger.info("Training complete!")
    
    if writer is not None:
        writer.close()
    
    if args.world_size > 1:
        dist.destroy_process_group()


def main():
    args = parse_args()
    
    if args.world_size > 1:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
    else:
        main_worker(0, args)


if __name__ == "__main__":
    main()
