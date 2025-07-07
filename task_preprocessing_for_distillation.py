#!/usr/bin/env python3
# coding: utf-8

import os
import gc
import sys
import logging
import torch
from pathlib import Path
from typing import Dict, Any
from blackbird.dataset import Dataset
from audio_separator import setup_models as setup_separator_models, process_single_file
from audio_pitch_rmvpe import setup_models as setup_rmvpe_models, extract_pitch
from tqdm import tqdm

# Set CUDA to only use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_track(track_path: str) -> None:
    """Process a single track for vocal separation and pitch extraction."""
    try:
        # Get the base directory and filename
        track_dir = os.path.dirname(track_path)
        track_base = os.path.basename(track_path)
        track_name = os.path.splitext(track_base)[0]
        
        # Define output paths in the same directory as the track
        vocal_output = os.path.join(track_dir, track_name + "_vocal.opus")
        dereverb_output = os.path.join(track_dir, track_name + "_dereverb.opus")
        pitch_output = os.path.join(track_dir, track_name + "_pitch.pt")
        
        # Skip if vocal and dereverb files exist
        if os.path.exists(vocal_output) and os.path.exists(dereverb_output):
            logger.info(f"Vocal and dereverb files already exist for {track_base}, skipping...")
            #return
        
        # Source file is the input track path
        source_file = track_path
        
        # Verify the file exists
        if not os.path.exists(source_file):
            logger.error(f"Source file not found: {source_file}")
            return
        
        # Step 1: Process audio separation
        #logger.info(f"Processing file: {source_file}")
        process_single_file(source_file, vocal_output=vocal_output, dereverb_output=dereverb_output)
        os.remove(vocal_output)
        
        # Step 2: Extract pitch from dereverb output
        if os.path.exists(dereverb_output) and not os.path.exists(pitch_output):
            logger.info(f"Extracting pitch from: {dereverb_output}")
            extract_pitch(
                audio_path=dereverb_output,
                output_path=pitch_output,
                model_name='rmvpe'
            )
            
    except Exception as e:
        logger.error(f"Error processing track {track_path}: {str(e)}")

def main():
    print("Loading dataset...")
    # Define the path to your dataset
    dataset_path = Path("/media/k4_nas/disk1/Datasets/Music_Part1")
   
    # Initialize the dataset
    logger.info(f"Initializing dataset at {dataset_path}")
    dataset = Dataset(dataset_path)
    
    # Get statistics after loading
    stats = dataset.analyze()
    
    # Print detailed statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total tracks: {stats['tracks']['total']}")
    logger.info(f"Complete tracks: {stats['tracks']['complete']}")
    
    logger.info("\nComponents:")
    for component, count in stats['components'].items():
        logger.info(f"- {component}: {count} files")

    # Initialize models for audio separation
    logger.info("\nInitializing audio separation models...")
    setup_separator_models({
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

    # Initialize RMVPE model
    logger.info("\nInitializing RMVPE model...")
    setup_rmvpe_models({
        'rmvpe': {
            'model_url': './ckpts/rmvpe.pt',
            'hop_length': 160,
            'threshold': 0.05
        }
    })

    # Get all artists and their albums
    artists_albums = dataset.index.album_by_artist
    logger.info(f"\nFound {len(artists_albums)} artists and their albums")

    # Print all artists
    print("All Artists:")
    
    # Get first 100 artists
    artists_to_process = list(stats["artists"])[:40]
    logger.info(f"\nProcessing first {len(artists_to_process)} artists")
    
    # Track all tracks to process
    tracks_to_process = []
    
    # Collect tracks for first 100 artists
    for artist in artists_to_process:

        # Print albums for this artist
        albums = stats["albums"][artist]
        for album in albums:
            track_dict = dataset.find_tracks(
                artist=artist, 
                album=album
            )

            # Process each track in the dictionary
            for track_files in track_dict.values():
                # Get the MP3 file path (should be first file with .mp3 extension)
                mp3_path = next((str(path) for path in track_files if str(path).endswith('.mp3')), None)
                if mp3_path:
                    tracks_to_process.append(mp3_path)

    logger.info(f"\nTotal tracks to process: {len(tracks_to_process)}")
    
    # Process all tracks
    for track_path in tqdm(tracks_to_process, desc="Processing tracks"):
        # clear gpu  
        torch.cuda.empty_cache()
        gc.collect()

        #logger.info(f"Processing track: {track_path}")
        process_track(track_path)

    logger.info("\nProcessing completed!")

if __name__ == "__main__":
    main() 