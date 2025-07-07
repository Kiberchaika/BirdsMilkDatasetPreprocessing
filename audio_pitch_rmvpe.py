#!/usr/bin/env python3
"""
Pitch extraction using RMVPE model
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import glob
import torch
import argparse
import numpy as np
import librosa
import time
from tqdm import tqdm
from RMVPE.src import RMVPE
import logging
from urllib.parse import quote

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for models
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODELS = {}

def download_file(url, path='ckpts'):
    """Download a file from a URL or copy from local path.
    
    Args:
        url: URL or local file path
        path: Target directory for downloaded files
        
    Returns:
        str: Path to the downloaded/copied file
    """
    # If input is a local file path
    if os.path.exists(url):
        return url
        
    # If it's a URL
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, os.path.basename(url))
    
    if os.path.exists(file_path):
        return file_path
    
    try:
        if url.startswith(('http://', 'https://')):
            torch.hub.download_url_to_file(quote(url, safe=':/'), file_path)
            return file_path
        else:
            raise ValueError(f"Invalid input: {url} - must be either a valid URL or existing local file path")
    except Exception as e:
        logger.error(f"Error accessing file: {e}")
        sys.exit(1)

def setup_models(model_configs):
    """Download and setup model files."""
    logger.info("Setting up models...")
      
    for model_name, config in model_configs.items():
        logger.info(f"Setting up {model_name} model...")
        model_path = download_file(config['model_url'])
        
        # Initialize RMVPE model with specified parameters
        model = RMVPE(model_path, hop_length=config.get('hop_length', 160))
        
        MODELS[model_name] = {
            'model': model,
            'hop_length': config.get('hop_length', 160),
            'threshold': config.get('threshold', 0.05)
        }
        logger.info(f"{model_name} model loaded successfully")

def extract_pitch(audio_path, output_path, model_name='rmvpe', device=None):
    """Extract pitch from audio file using RMVPE and save as PT file."""
    
    if device is None:
        device = DEVICE
    
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Please set up models first.")
    
    model_info = MODELS[model_name]
    model = model_info['model']
    hop_length = model_info['hop_length']
    threshold = model_info['threshold']
    
    logger.info(f'Loading audio for {os.path.basename(audio_path)}')
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    logger.info('Starting inference...')
    t = time.time()
    
    # Extract pitch using RMVPE
    f0 = model.infer_from_audio(audio, int(sr), device=device, thred=threshold, use_viterbi=False)
    
    infer_time = time.time() - t
    logger.info(f'Inference time: {infer_time:.3f}s')
    logger.info(f'RTF: {infer_time * sr / len(audio):.3f}')
    
    # Prepare pitch data
    pitch_data = {
        'pitch': f0,
        'sample_rate': sr,
        'hop_length': hop_length,
        'threshold': threshold,
        'inference_time': infer_time,
        'rtf': infer_time * sr / len(audio)
    }
    
    # Save data as PT file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(pitch_data, output_path)
    
    logger.info(f"Saved: {output_path}")
    return pitch_data

def main():
    parser = argparse.ArgumentParser(description='Extract pitch from audio files using RMVPE')
    parser.add_argument('-i', '--input', type=str, default='./input', 
                       help='Input file or directory')
    parser.add_argument('-d', '--device', type=str, default=None, 
                       help='Device to use (cpu/cuda), auto-detected if not set')
    parser.add_argument('-hop', '--hop_length', type=int, default=160, 
                       help='Hop length under 16kHz sampling rate (default: 160)')
    parser.add_argument('-th', '--threshold', type=float, default=0.05, 
                       help='Unvoice threshold (default: 0.05)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = DEVICE
    else:
        device = args.device
    
    logger.info(f'Using device: {device}')
    
    # Setup models
    setup_models({
        'rmvpe': {
            'model_url': './ckpts/rmvpe.pt',
            'hop_length': args.hop_length,
            'threshold': args.threshold
        }
    })
    
    # Handle single file or directory
    if os.path.isfile(args.input):
        files = [args.input]
    else:
        files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.opus', '*.m4a']:
            files.extend(glob.glob(os.path.join(args.input, ext)))
    
    if not files:
        logger.error("No audio files found!")
        return
    
    logger.info(f"Found {len(files)} audio files to process")
    
    # Process files
    for audio_path in tqdm(files, desc="Processing"):
        try:
            base_name = os.path.splitext(audio_path)[0]
            output_path = f"{base_name}_pitch_rmvpe.pt"
            
            extract_pitch(
                audio_path=audio_path,
                output_path=output_path,
                model_name='rmvpe',
                device=device
            )
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(audio_path)}: {e}")

if __name__ == "__main__":
    main()