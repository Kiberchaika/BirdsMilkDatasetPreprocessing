# https://bascurtiz.x10.mx/models-checkpoint-config-urls.html
# 

import os
import logging
import subprocess
from blackbird.dataset import Dataset
from pathlib import Path
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_convert_to_opus(track_info: Dict[str, Any]) -> None:
    """Convert source music track to opus format."""
    source_file = track_info.get("source_music_track")
    if not source_file:
        logger.error(f"No source music track found for {track_info['track_path']}")
        return
    
    # Check if source file exists before proceeding
    if not os.path.exists(source_file):
        logger.error(f"Source file does not exist: {source_file}")
        # Try to find the file with similar name in the directory
        dir_path = os.path.dirname(source_file)
        if os.path.exists(dir_path):
            logger.info(f"Searching for similar files in {dir_path}")
            files = os.listdir(dir_path)
            base_name = os.path.basename(source_file)
            logger.info(f"Available files: {files}")
            logger.info(f"Looking for file similar to: {base_name}")
        return
    
    # replace extension with .opus
    output_file = source_file.replace(".mp3", ".opus")
    logger.info(f"Converting {source_file} to opus format...")
    
    try:
        # Use ffmpeg to convert to opus format with optimized settings
        cmd = [
            "ffmpeg", "-i", source_file,
            "-y",
            "-c:a", "libopus",
            "-b:a", "192k",
            "-vbr", "on",
            "-compression_level", "10",
            "-map_metadata", "0",
            output_file
        ]
        # Run the command and capture output
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        # Check if the command was successful
        if result.returncode != 0:
            logger.error(f"Error converting to opus. Return code: {result.returncode}")
            logger.error(f"Error details: {result.stderr}")
            return
        
        logger.info(f"Successfully converted to opus: {output_file}")
    except Exception as e:
        logger.error(f"Exception during conversion: {str(e)}")

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
    #dataset.rebuild_index()
    
    # Print detailed statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total tracks: {stats['tracks']['total']}")
    logger.info(f"Complete tracks: {stats['tracks']['complete']}")
    
    logger.info("\nComponents:")
    for component, count in stats['components'].items():
        logger.info(f"- {component}: {count} files")
     
    # Example of finding tracks with specific components
    tracks_with_all = dataset.find_tracks()
    logger.info(f"\nFound {len(tracks_with_all)} total tracks")
    
    # Print schema components to understand patterns
    logger.info("\nSchema components:")
    for name, config in dataset._schema.schema["components"].items():
        logger.info(f"- {name}: {config}")
    
    # 1. Find tracks without source_music_track_opus and convert them
    logger.info("\nFinding tracks without opus format...")
    tracks_without_opus = dataset.find_tracks(missing=["source_music_track_opus"])
    logger.info(f"Found {len(tracks_without_opus)} tracks without opus format")
    
    for track_path in tracks_without_opus:
        logger.info(f"\nProcessing {track_path}")
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
            
            track_info = {
                "track_path": track_path,
                "source_music_track": str(source_file)
            }
            preprocess_convert_to_opus(track_info)
        except Exception as e:
            logger.error(f"Error processing track {track_path}: {str(e)}")
    

    os.exit(0)
    
    # 2. Find tracks without vocal_separated and separate vocals
    logger.info("\nFinding tracks without separated vocals...")
    tracks_without_vocals = dataset.find_tracks(missing=["vocal_separated"])
    logger.info(f"Found {len(tracks_without_vocals)} tracks without separated vocals")
    
    for track_path in tracks_without_vocals:
        logger.info(f"\nProcessing {track_path}")
        try:
            # Get the base directory and filename
            track_dir = os.path.dirname(track_path)
            track_base = os.path.basename(track_path)
            
            # Construct the source file path
            source_file = os.path.join(dataset_path, track_dir, track_base + ".opus")
            
            # Verify the file exists
            if not os.path.exists(source_file):
                logger.warning(f"Opus file not found: {source_file}")
                # Try to find the actual file in the directory
                dir_path = os.path.join(dataset_path, track_dir)
                if os.path.exists(dir_path):
                    opus_files = [f for f in os.listdir(dir_path) if f.endswith('.opus')]
                    if opus_files:
                        # Find the most similar filename
                        logger.info(f"Found {len(opus_files)} opus files in directory")
                        # Use the first opus file as a fallback
                        source_file = os.path.join(dir_path, opus_files[0])
                        logger.info(f"Using alternative file: {source_file}")
            
            track_info = {
                "track_path": track_path,
                "source_music_track_opus": str(source_file)
            }
            preprocess_separate_vocal(track_info)
        except Exception as e:
            logger.error(f"Error processing track {track_path}: {str(e)}")
    
    # 3. Find tracks with vocals but without dereverb and process them
    logger.info("\nFinding tracks without dereverberated vocals...")
    tracks_without_dereverb = dataset.find_tracks(
        has=["vocal_separated"],
        missing=["vocal_separated_dereverb"]
    )
    logger.info(f"Found {len(tracks_without_dereverb)} tracks without dereverberated vocals")
    
    for track_path in tracks_without_dereverb:
        logger.info(f"\nProcessing {track_path}")
        try:
            # Get the base directory and filename
            track_dir = os.path.dirname(track_path)
            track_base = os.path.basename(track_path)
            
            # Construct the vocal file path
            vocal_file = os.path.join(dataset_path, track_dir, track_base + "_voc.opus")
            
            # Verify the file exists
            if not os.path.exists(vocal_file):
                logger.warning(f"Vocal file not found: {vocal_file}")
                # Try to find the actual file in the directory
                dir_path = os.path.join(dataset_path, track_dir)
                if os.path.exists(dir_path):
                    voc_files = [f for f in os.listdir(dir_path) if f.endswith('_voc.opus')]
                    if voc_files:
                        # Find the most similar filename
                        logger.info(f"Found {len(voc_files)} vocal files in directory")
                        # Use the first vocal file as a fallback
                        vocal_file = os.path.join(dir_path, voc_files[0])
                        logger.info(f"Using alternative file: {vocal_file}")
            
            track_info = {
                "track_path": track_path,
                "vocal_separated": str(vocal_file)
            }
            preprocess_dereverb(track_info)
        except Exception as e:
            logger.error(f"Error processing track {track_path}: {str(e)}")
    
    logger.info("\nPreprocessing completed!")

if __name__ == "__main__":
    main() 