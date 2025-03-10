# https://bascurtiz.x10.mx/models-checkpoint-config-urls.html
# 

import os
import logging
import subprocess
from blackbird.dataset import Dataset
from pathlib import Path
from typing import Dict, Any
from audio_separator import setup_models, process_single_file

# Set CUDA to only use GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_separate_vocal(track_info: Dict[str, Any]) -> None:
    """Separate vocals from the music track using Demucs."""
    source_file = track_info.get("source_music_track_opus")
    if not source_file:
        logger.error(f"No opus source track found for {track_info['track_path']}")
        return
    
    # Check if source file exists before proceeding
    if not os.path.exists(source_file):
        logger.error(f"Source opus file does not exist: {source_file}")
        # Try to find the file with similar name in the directory
        dir_path = os.path.dirname(source_file)
        if os.path.exists(dir_path):
            logger.info(f"Searching for similar files in {dir_path}")
            files = os.listdir(dir_path)
            base_name = os.path.basename(source_file)
            logger.info(f"Available files: {files}")
            logger.info(f"Looking for file similar to: {base_name}")
        return
    
    output_dir = os.path.dirname(source_file)
    logger.info(f"Separating vocals from {source_file}...")
    
    try:
        # Use demucs to separate vocals
        cmd = [
            "demucs", "--two-stems=vocals",
            "-n", "htdemucs",
            "--out", output_dir,
            source_file
        ]
        # Run the command and capture output
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        # Check if the command was successful
        if result.returncode != 0:
            logger.error(f"Error separating vocals. Return code: {result.returncode}")
            logger.error(f"Error details: {result.stderr}")
            return
            
        logger.info(f"Successfully separated vocals for: {source_file}")
    except Exception as e:
        logger.error(f"Exception during vocal separation: {str(e)}")

def preprocess_dereverb(track_info: Dict[str, Any]) -> None:
    """Remove reverb from separated vocals using RVC."""
    vocal_file = track_info.get("vocal_separated")
    if not vocal_file:
        logger.error(f"No separated vocals found for {track_info['track_path']}")
        return
    
    # Check if vocal file exists before proceeding
    if not os.path.exists(vocal_file):
        logger.error(f"Vocal file does not exist: {vocal_file}")
        # Try to find the file with similar name in the directory
        dir_path = os.path.dirname(vocal_file)
        if os.path.exists(dir_path):
            logger.info(f"Searching for similar files in {dir_path}")
            files = os.listdir(dir_path)
            base_name = os.path.basename(vocal_file)
            logger.info(f"Available files: {files}")
            logger.info(f"Looking for file similar to: {base_name}")
        return
    
    output_file = vocal_file.replace("_voc.opus", "_voc_der.opus")
    logger.info(f"Removing reverb from {vocal_file}...")
    
    try:
        # Use RVC to remove reverb
        cmd = [
            "rvc", "dereverb",
            "--input", vocal_file,
            "--output", output_file
        ]
        # Run the command and capture output
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        # Check if the command was successful
        if result.returncode != 0:
            logger.error(f"Error removing reverb. Return code: {result.returncode}")
            logger.error(f"Error details: {result.stderr}")
            return
            
        logger.info(f"Successfully removed reverb: {output_file}")
    except Exception as e:
        logger.error(f"Exception during dereverb: {str(e)}")

def main():
    print("Loading dataset...")
    # Define the path to your dataset
    dataset_path = Path("/media/k4_nas/disk1/Datasets/Music/FUNK")
   
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

    # Initialize models
    logger.info("\nInitializing models...")
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

    # Get all tracks
    tracks = dataset.find_tracks()
    logger.info(f"\nFound {len(tracks)} total tracks")

    # Group tracks by album
    albums = {}
    for track_path in tracks:
        album_path = os.path.dirname(track_path)
        if album_path not in albums:
            albums[album_path] = []
        albums[album_path].append(track_path)

    # Process first track from each album
    for album_path, album_tracks in albums.items():
        if not album_tracks:
            continue
            
        track_path = album_tracks[0]  # Get first track
        logger.info(f"\nProcessing first track from album: {album_path}")
        logger.info(f"Track: {track_path}")
        
        try:
            # Get the base directory and filename
            track_dir = os.path.dirname(track_path)
            track_base = os.path.basename(track_path)
            
            # Construct the source file path
            source_file = os.path.join(dataset_path, track_dir, track_base + ".mp3")
            
            # Verify the file exists
            if not os.path.exists(source_file):
                logger.warning(f"Source file not found: {source_file}")
                # Try to find the actual file in the directory
                dir_path = os.path.join(dataset_path, track_dir)
                if os.path.exists(dir_path):
                    mp3_files = [f for f in os.listdir(dir_path) if f.endswith('.mp3')]
                    if mp3_files:
                        # Find the most similar filename
                        logger.info(f"Found {len(mp3_files)} MP3 files in directory")
                        # Use the first MP3 file as a fallback
                        source_file = os.path.join(dir_path, mp3_files[0])
                        logger.info(f"Using alternative file: {source_file}")
                else:
                    logger.error(f"Directory not found: {dir_path}")
                    continue
            
            # Process the track
            logger.info(f"Processing file: {source_file}")
            process_single_file(source_file, "vocal1", "vocal1_dereverb")
            
        except Exception as e:
            logger.error(f"Error processing track {track_path}: {str(e)}")
            continue

    logger.info("\nProcessing completed!")

if __name__ == "__main__":
    main() 