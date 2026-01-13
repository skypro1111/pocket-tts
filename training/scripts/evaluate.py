"""Evaluation utilities for finetuned models."""

import argparse
import json
import logging
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    model: TTSModel,
    test_data: list[dict],
    output_dir: Path,
    voice_prompt: str = None,
    max_samples: int = None,
):
    """Evaluate model on test data.
    
    Args:
        model: TTS model to evaluate
        test_data: List of test samples with 'text' and 'language'
        output_dir: Directory to save generated audio
        voice_prompt: Voice prompt audio file
        max_samples: Maximum number of samples to evaluate
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get voice state
    if voice_prompt is None:
        voice_prompt = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
    
    voice_state = model.get_state_for_audio_prompt(voice_prompt)
    
    # Evaluate samples
    results = []
    
    samples_to_eval = test_data[:max_samples] if max_samples else test_data
    
    for i, sample in enumerate(tqdm(samples_to_eval, desc="Evaluating")):
        text = sample["text"]
        language = sample.get("language", "unknown")
        
        try:
            # Generate audio
            audio = model.generate_audio(voice_state, text)
            
            # Save audio
            output_path = output_dir / f"sample_{i:04d}_{language}.wav"
            torchaudio.save(
                output_path,
                audio.unsqueeze(0).cpu(),
                model.sample_rate,
            )
            
            results.append({
                "index": i,
                "text": text,
                "language": language,
                "output_file": str(output_path),
                "success": True,
            })
            
        except Exception as e:
            logger.error(f"Failed to generate sample {i}: {e}")
            results.append({
                "index": i,
                "text": text,
                "language": language,
                "success": False,
                "error": str(e),
            })
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"Successfully generated {success_count}/{len(results)} samples")
    logger.info(f"Results saved to {results_file}")


def load_model_from_checkpoint(
    config_path: str,
    checkpoint_path: str,
) -> TTSModel:
    """Load model from checkpoint."""
    config = load_config(config_path)
    
    model = TTSModel._from_pydantic_config(
        config,
        temp=1.0,
        lsd_decode_steps=4,
        noise_clamp=None,
        eos_threshold=0.5
    )
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned TTS model")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pocket_tts/config/b6369a24.yaml",
        help="Path to model config"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory for generated audio"
    )
    parser.add_argument(
        "--voice_prompt",
        type=str,
        default=None,
        help="Voice prompt audio file"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.config, args.checkpoint)
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}...")
    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Evaluate
    output_dir = Path(args.output_dir)
    evaluate_model(
        model,
        test_data,
        output_dir,
        args.voice_prompt,
        args.max_samples,
    )


if __name__ == "__main__":
    main()
