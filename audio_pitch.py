#!/usr/bin/env python3

"""
Simplified pitch extraction using torchcrepe
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torchcrepe
import librosa
from torch.amp.autocast_mode import autocast

def extract_pitch(audio_path, output_path, fmin=50, fmax=550, model='full', hop_length_ms=10):
    """Extract pitch from audio file and save as PT file."""
    
    # Load audio with 16kHz sampling rate use librosa and convert to torch tensor 
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio = torch.from_numpy(audio).float()
    print(f"[DEBUG] audio.shape after loading: {audio.shape}, type: {type(audio)}")
    
    # Add batch dimension if audio is 1D
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
        print(f"[DEBUG] audio.shape after unsqueeze: {audio.shape}")
    
    # Extract pitch using torchcrepe
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[DEBUG] device: {device}")
    print(f"[DEBUG] audio.shape before torchcrepe: {audio.shape}, audio.ndim: {audio.ndim}")
    if device == 'cuda':
        with autocast('cuda', enabled=True):
            pitch = torchcrepe.predict(
                audio,
                sr,
                hop_length=160,
                fmin=fmin,
                fmax=fmax,
                model=model,
                batch_size=2048,
                device=device
            )
    else:
        pitch = torchcrepe.predict(
            audio,
            sr,
            hop_length=160,
            fmin=fmin,
            fmax=fmax,
            model=model,
            batch_size=2048,
            device=device
        )
        
    # Filter and smooth
    pitch = torchcrepe.filter.median(pitch, 3)
    periodicity = torchcrepe.predict(
        audio, sr, hop_length=160,
        fmin=fmin, fmax=fmax, model=model, return_periodicity=True,
        batch_size=2048, device=device
    )[1]
    pitch = torchcrepe.threshold.At(0.21)(pitch, periodicity)
    
    # Save data
    pitch_data = {
        'pitch': pitch.cpu().numpy() if torch.is_tensor(pitch) else pitch,
        'sample_rate': sr,
        'hop_length_ms': hop_length_ms,
        'fmin': fmin,
        'fmax': fmax
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(pitch_data, output_path)
    
    print(f"Saved: {output_path}")
    return pitch_data

def main():
    parser = argparse.ArgumentParser(description='Extract pitch from audio files')
    parser.add_argument('--input', default='./input', help='Input file or directory')
    parser.add_argument('--fmin', default=50, type=float, help='Min frequency (Hz)')
    parser.add_argument('--fmax', default=550, type=float, help='Max frequency (Hz)')
    parser.add_argument('--model', default='full', choices=['tiny', 'full'], help='Model size')
    parser.add_argument('--hop_ms', default=10, type=float, help='Hop length (ms)')
    
    args = parser.parse_args()
    
    # Handle single file or directory
    if os.path.isfile(args.input):
        files = [args.input]
    else:
        files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.opus', '*.m4a']:
            files.extend(glob.glob(os.path.join(args.input, ext)))
    
    if not files:
        print("No audio files found!")
        return
    
    # Process files
    for audio_path in tqdm(files, desc="Processing"):
        try:
            base_name = os.path.splitext(audio_path)[0]
            output_path = f"{base_name}_pitch_crepe.pt"
            
            extract_pitch(
                audio_path, output_path,
                fmin=args.fmin, fmax=args.fmax,
                model=args.model, hop_length_ms=args.hop_ms
            )
        except Exception as e:
            print(f"Error with {os.path.basename(audio_path)}: {e}")

if __name__ == "__main__":
    main()