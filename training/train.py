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
    
    return parser.parse_args()


def compute_loss(model, batch, device):
    """Compute training loss for a batch.
    
    This is a simplified loss computation. In practice, you would:
    1. Encode audio with Mimi encoder
    2. Pass text and encoded audio through FlowLM
    3. Compute flow matching loss
    """
    audios = batch["audios"].to(device)
    texts = batch["texts"]
    audio_lengths = batch["audio_lengths"].to(device)
    
    # For demonstration, we compute a simple reconstruction loss
    # In actual training, this would involve:
    # - Encoding audio to latents with Mimi
    # - Computing flow matching loss with FlowLM
    # - Possibly add speaker embedding loss
    
    # Placeholder loss computation
    # TODO: Implement actual flow matching loss
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    logger.warning(
        "Using placeholder loss! Implement actual flow matching loss for real training."
    )
    
    return loss


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    epoch,
    args,
    device,
    writer=None,
    global_step=0,
):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    epoch_start = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        # Compute loss
        loss = compute_loss(model, batch, device)
        
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
        
        # Logging
        if (batch_idx + 1) % args.log_interval == 0:
            lr = get_lr(optimizer)
            elapsed = time.time() - epoch_start
            
            logger.info(
                f"Epoch [{epoch}] Step [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss_meter.avg:.4f} LR: {lr:.6f} "
                f"Time: {format_time(elapsed)}"
            )
            
            if writer is not None:
                writer.add_scalar("train/loss", loss_meter.avg, global_step)
                writer.add_scalar("train/lr", lr, global_step)
        
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
            if hasattr(param, "requires_grad"):
                param.requires_grad = False
    
    model = model.to(device)
    
    # Wrap with DDP if distributed
    if args.world_size > 1:
        model = DDP(model, device_ids=[gpu])
    
    logger.info(f"Model has {count_parameters(model):,} trainable parameters")
    
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
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
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
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            epoch,
            args,
            device,
            writer,
            global_step,
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
