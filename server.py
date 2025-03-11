from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
from blackbird.dataset import Dataset
from tqdm import tqdm
import math

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Global variables
dataset = None
compositions = []

def initialize_dataset() -> None:
    """Initialize the blackbird dataset and scan for compositions."""
    global dataset, compositions
    try:
        dataset_path = Path("/media/k4_nas/disk1/Datasets/Music/FUNK")
        logger.info(f"Initializing dataset at {dataset_path}")
        dataset = Dataset(dataset_path)
        
        # Get statistics after loading
        stats = dataset.analyze()
        logger.info("\nDataset Statistics:")
        logger.info(f"Total tracks: {stats['tracks']['total']}")
        logger.info(f"Complete tracks: {stats['tracks']['complete']}")
        
        # Get all tracks and build compositions
        tracks = dataset.find_tracks()
        logger.info(f"\nFound {len(tracks)} total tracks")
        
        # Group tracks by album with progress bar
        albums = {}
        for track_path in tqdm(tracks, desc="Grouping tracks by album"):
            album_path = os.path.dirname(track_path)
            if album_path not in albums:
                albums[album_path] = []
            albums[album_path].append(track_path)
        
        # Convert albums to compositions format
        compositions_list = []
        for album_path, album_tracks in tqdm(albums.items(), desc="Processing albums"):
            if not album_tracks:
                continue
            
            album_name = os.path.basename(album_path)
            composition = {
                "id": len(compositions_list) + 1,
                "title": album_name,
                "tracks": []
            }
            
            # Process first track from album
            track_path = album_tracks[0]
            track_dir = os.path.dirname(track_path)
            track_base = os.path.basename(track_path)
            
            main_track_path = os.path.join(dataset_path, track_dir, track_base + ".opus")
            vocal1_path = os.path.join(dataset_path, track_dir, track_base + "_vocal1.mp3")
            vocal1_dereverb_path = os.path.join(dataset_path, track_dir, track_base + "_vocal1_dereverb.mp3")
            vocal2_path = os.path.join(dataset_path, track_dir, track_base + "_vocal2.mp3")
            vocal2_dereverb_path = os.path.join(dataset_path, track_dir, track_base + "_vocal2_dereverb.mp3")
            
            if os.path.exists(main_track_path) and os.path.exists(vocal1_path) and os.path.exists(vocal1_dereverb_path) and os.path.exists(vocal2_path) and os.path.exists(vocal2_dereverb_path):
                composition["tracks"] = [
                    {
                        "title": "Main Track",
                        "type": "main",
                        "url": f"http://localhost:3000/audio/{os.path.relpath(main_track_path, dataset_path)}",
                        "markers": []
                    },
                    {
                        "title": "Vocal 1",
                        "type": "vocal1",
                        "url": f"http://localhost:3000/audio/{os.path.relpath(vocal1_path, dataset_path)}",
                        "markers": []
                    },
                    {
                        "title": "Vocal 2",
                        "type": "vocal2",
                        "url": f"http://localhost:3000/audio/{os.path.relpath(vocal2_path, dataset_path)}",
                        "markers": []
                    },
                    {
                        "title": "Vocal 1 (Dereverb)",
                        "type": "vocal1_dereverb",
                        "url": f"http://localhost:3000/audio/{os.path.relpath(vocal1_dereverb_path, dataset_path)}",
                        "markers": []
                    },
                    {
                        "title": "Vocal 2 (Dereverb)",
                        "type": "vocal2_dereverb",
                        "url": f"http://localhost:3000/audio/{os.path.relpath(vocal2_dereverb_path, dataset_path)}",
                        "markers": []
                    }
                ]
                compositions_list.append(composition)
        
        compositions = compositions_list
        logger.info(f"Successfully loaded {len(compositions)} compositions")
        
    except Exception as error:
        logger.error(f'Error initializing dataset: {str(error)}')
        compositions = []

# Initialize dataset on startup
initialize_dataset()

@app.route('/api/compositions')
def get_compositions():
    # Get pagination parameters from query string
    page = request.args.get('page', default=1, type=int)
    limit = request.args.get('limit', default=5, type=int)
    
    # Calculate start and end indices
    start_index = (page - 1) * limit
    end_index = page * limit
    
    # Get paginated compositions
    paginated_compositions = compositions[start_index:end_index]
    
    # Prepare response with pagination metadata
    response = {
        "compositions": paginated_compositions,
        "pagination": {
            "total": len(compositions),
            "currentPage": page,
            "totalPages": math.ceil(len(compositions) / limit),
            "hasNextPage": end_index < len(compositions),
            "hasPrevPage": page > 1
        }
    }
    
    return jsonify(response)

@app.route('/api/compositions/<int:composition_id>')
def get_composition(composition_id):
    composition = next((c for c in compositions if c["id"] == composition_id), None)
    if not composition:
        return jsonify({"error": "Composition not found"}), 404
    return jsonify(composition)

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    dataset_path = Path("/media/k4_nas/disk1/Datasets/Music/FUNK")
    return send_from_directory(str(dataset_path), filename)

@app.route('/api/rescan', methods=['POST'])
def rescan():
    try:
        initialize_dataset()
        return jsonify({
            "message": "Dataset rescanned successfully",
            "count": len(compositions)
        })
    except Exception as error:
        return jsonify({"error": f"Failed to rescan dataset: {str(error)}"}), 500

if __name__ == '__main__':
    app.run(port=3000) 