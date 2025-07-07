#!/usr/bin/env python3
# coding: utf-8

# https://bascurtiz.x10.mx/models-checkpoint-config-urls.html

"""
Inference script for vocal extraction and de-reverberation 
"""

import os
import sys
import time
import glob
import torch
import librosa
import argparse
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
from urllib.parse import quote
import warnings
import subprocess
import logging

# Add current directory to path for imports
sys.path.append('./Music-Source-Separation-Training')

from utils import get_model_from_config, demix

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for models and configs
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
        config_path = download_file(config['config_url'])
        ckpt_path = download_file(config['ckpt_url'])
        
        model, model_config = get_model_from_config(config['model_type'], config_path)
        
        # Override chunk_size and batch_size in config
        if hasattr(model_config.inference, 'chunk_size'):
            model_config.inference.chunk_size *= 5
        if hasattr(model_config.inference, 'batch_size'):
            # Increase batch size based on available VRAM
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb >= 24:  # High-end GPUs (3090/4090)
                    model_config.inference.batch_size = 8
                elif vram_gb >= 16:  # Mid-range GPUs
                    model_config.inference.batch_size = 6
                elif vram_gb >= 8:  # Low-end GPUs
                    model_config.inference.batch_size = 4
                else:
                    model_config.inference.batch_size = 2
            else:
                model_config.inference.batch_size = 2
                
        # Optimize overlap for faster processing while maintaining quality
        if hasattr(model_config.inference, 'num_overlap'):
            model_config.inference.num_overlap = 2
            
        # Enable mixed precision for faster inference
        if hasattr(model_config.training, 'use_amp'):
            model_config.training.use_amp = True
        
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
        MODELS[model_name] = {'model': model.eval().to(DEVICE), 'config': model_config, 'model_type': config['model_type']}
        logger.info(f"{model_name} model loaded successfully")

def normalize_audio(audio):
    """Normalize audio using mean and standard deviation.
    
    Args:
        audio: Input audio array of shape (channels, samples)
        
    Returns:
        tuple: (normalized audio, (mean, std) for denormalization)
    """
    if len(audio.shape) == 1:  # Handle mono
        mono = audio
    else:  # Handle stereo by taking mean across channels
        mono = audio.mean(0)
    
    mean = mono.mean()
    std = mono.std()
    
    # Avoid division by zero
    if std == 0:
        std = 1e-10
        
    normalized = (audio - mean) / std
    return normalized, (mean, std)

def denormalize_audio(audio, norm_params):
    """Denormalize audio using stored mean and standard deviation.
    
    Args:
        audio: Normalized audio array
        norm_params: Tuple of (mean, std) from normalization
    """
    mean, std = norm_params
    return (audio * std) + mean

def process_audio(model_info, audio_path, output_path, extract_instrumental=False):
    """Process audio with the given model."""
 
    model, config = model_info['model'], model_info['config']
    model_type = model_info['model_type']
    sample_rate = getattr(config.audio, 'sample_rate', 44100)
    
    try:
        # Load audio file with resampling if needed
        mix, orig_sr = librosa.load(audio_path, sr=None, mono=False)
        if orig_sr != sample_rate:
            mix = librosa.resample(mix, orig_sr=orig_sr, target_sr=sample_rate)
    except Exception as e:
        logger.error(f'Cannot read track: {audio_path}')
        logger.error(f'Error message: {str(e)}')
        raise
    
    # Handle mono/stereo conversion based on model requirements
    if len(mix.shape) == 1:  # Mono audio
        mix = np.expand_dims(mix, axis=0)
        if 'num_channels' in config.audio:
            if config.audio['num_channels'] == 2:
                mix = np.concatenate([mix, mix], axis=0)

    elif len(mix.shape) == 2 and mix.shape[0]>1: # Stereo
        mix = np.mean(mix, 0 , keepdims = True)
        if 'num_channels' in config.audio:
            if config.audio['num_channels'] == 2:
                mix = np.concatenate([mix, mix], axis=0)

        
    
    # Store original mix for later use
    mix_orig = mix.copy()
    
    # Use mixed precision for faster processing
    with torch.amp.autocast('cuda', enabled=getattr(config.training, 'use_amp', True)):
        # Normalize if required by config
        if getattr(config.inference, 'normalize', False):
            mix, norm_params = normalize_audio(mix)
        
        # Convert to tensor and process through model
        try:
            waveforms = demix(config, model, mix, DEVICE, model_type=model_type, pbar=True)
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            raise
    
    # Get the appropriate output key based on model type
    instruments = config.training.instruments
    key = instruments[0] if instruments else 'vocals'
    if key not in waveforms:
        raise KeyError(f"No valid output key found in waveforms. Expected {key}")
    
    estimates = waveforms[key]
    
    # Denormalize if necessary
    if getattr(config.inference, 'normalize', False):
        estimates = denormalize_audio(estimates, norm_params)
    
    # Handle instrumental extraction if requested
    if extract_instrumental:
        instrumental = mix_orig - estimates
        if output_path:
            instr_path = output_path.replace('vocals', 'instrumental')
            os.makedirs(os.path.dirname(instr_path), exist_ok=True)
            try:
                sf.write(instr_path, instrumental.T, sample_rate, subtype='PCM_24')
            except Exception as e:
                logger.error(f"Error saving instrumental file: {e}")
                raise
    
    # Save the processed audio
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            sf.write(output_path, estimates.T, sample_rate, subtype='PCM_24')
        except Exception as e:
            logger.error(f"Error saving output file: {e}")
            raise
    
    return estimates

def process_single_file(audio_path, vocal_output, dereverb_output):
    """Process a single audio file for vocal extraction and de-reverberation.
    
    Args:
        audio_path (str): Path to the input audio file
        vocal_output (str): Path for the vocal extraction output (opus format)
        dereverb_output (str): Path for the de-reverb output (opus format)
    """
    start_time = time.time()
    
    # Process vocal extraction
    temp_vocal = vocal_output.replace('.opus', '_temp.wav')
    process_audio(
        MODELS['vocal'],
        audio_path,
        temp_vocal,
        extract_instrumental=False
    )
    
    # Convert vocal to opus format with high quality
    subprocess.run(['ffmpeg', '-y', '-i', temp_vocal, '-c:a', 'libopus', '-b:a', '192k', vocal_output], capture_output=True)
    os.remove(temp_vocal)
    
    # Process dereverb
    temp_dereverb = dereverb_output.replace('.opus', '_temp.wav')
    process_audio(
        MODELS['dereverb'],
        vocal_output,
        temp_dereverb,
        extract_instrumental=False
    )
    
    # Convert dereverb to opus format with high quality
    subprocess.run(['ffmpeg', '-y', '-i', temp_dereverb, '-c:a', 'libopus', '-b:a', '192k', dereverb_output], capture_output=True)
    os.remove(temp_dereverb)
    
    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
    return dereverb_output

def main():
    parser = argparse.ArgumentParser(
        description='Run vocal extraction and de-reverberation on opus files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir', 
        default='./input', 
        type=str, 
        help='Directory containing opus files to process'
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' not found")
        sys.exit(1)
    
    opus_files = glob.glob(os.path.join(args.input_dir, '*.opus'))
    if not opus_files:
        print(f"No opus files found in '{args.input_dir}'")
        sys.exit(1)
    
    # Load models once at the start
    print("Loading models...")
    '''
    setup_models({
        'vocal': {
            'config_url': 'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/refs/heads/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml',
            'ckpt_url': 'https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt',
            'model_type': 'mel_band_roformer'
        },
        'dereverb': {
            'config_url': '/home/k4/Python/Music-Source-Separation-Training/dereverb_mel_band_roformer_anvuew.yaml',
            'ckpt_url': '/home/k4/Python/Music-Source-Separation-Training/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt',
            'model_type': 'mel_band_roformer'
        }
    })
    '''

    setup_models({
        'vocal': {
            'config_url': 'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml',
            'ckpt_url': 'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft.ckpt',
            'model_type': 'mel_band_roformer'
        },
        'dereverb': {
            'config_url': 'https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml',
            'ckpt_url': 'https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt',
            'model_type': 'mdx23c'
        }
    })

    print(f"Processing {len(opus_files)} files...")
    for audio_path in tqdm(opus_files):
        try:
            vocal_output = os.path.join(os.path.dirname(audio_path), os.path.splitext(os.path.basename(audio_path))[0] + '_vocal.opus')
            dereverb_output = os.path.join(os.path.dirname(audio_path), os.path.splitext(os.path.basename(audio_path))[0] + '_dereverb.opus')
            process_single_file(
                audio_path,
                vocal_output,
                dereverb_output
            )
            os.remove(vocal_output)
        except Exception as e:
            print(f"Error processing {os.path.basename(audio_path)}: {e}")
    
    print("Done!")

if __name__ == "__main__":
    main()