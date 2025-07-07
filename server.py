from flask import Flask, jsonify, send_from_directory, request, Response, send_file
from flask_cors import CORS
import os
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
from blackbird.dataset import Dataset
from tqdm import tqdm
import math
import socket
import mimetypes
from text_compare import get_diff

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Global variables
dataset = None
compositions = []
HOST = '0.0.0.0'  # This will allow external connections
PORT = 7779

@app.route('/')
def index():
    return "Server is running! 🚀"

def get_server_url():
    # Get the actual network IP address
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable
        s.connect(('10.255.255.255', 1))
        actual_ip = s.getsockname()[0]
    except Exception:
        actual_ip = '127.0.0.1'
    finally:
        s.close()
    return f"http://{actual_ip}:{PORT}"


"""Process text differences and add HTML spans for highlighting."""
def highlight_diffs(text1, text2, text3):
    # Use get_diff which takes a list of texts
    diff_result = get_diff([text1, text2, text3])
    
    # Process each text's diff results
    html_texts = []
    
    for text_result in diff_result:
        processed_text = []
        for char, presence in text_result:
            is_unique = len(presence) == 1  # Unique to this text if only present in one text
            
            if not is_unique:
                if char == '\n':
                    processed_text.append(char + '<br>')
                elif char == ' ':
                    processed_text.append('&nbsp;')
                else:
                    processed_text.append(char)
            else:
                style_attr = ' style="background: #ffebee;"'
                
                if char == '\n':
                    processed_text.append(f'<span{style_attr}>{char}</span><br>')
                elif char == ' ':
                    processed_text.append(f'<span{style_attr}>&nbsp;</span>')
                else:
                    processed_text.append(f'<span{style_attr}>{char}</span>')
        
        html_texts.append(''.join(processed_text))
    
    return html_texts

def initialize_dataset() -> None:
    """Initialize the blackbird dataset and scan for compositions."""
    global dataset, compositions
    try:
        dataset_path = Path("/media/k4_nas/disk1/Datasets/Music_Part1")
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
        for track_name, track_paths in tqdm(tracks.items(), desc="Grouping tracks by album"):
            for track_path in track_paths:
                track_path = os.path.relpath(track_path, dataset_path)
                album_path = os.path.dirname(track_path)
                if album_path not in albums:
                    albums[album_path] = []
                albums[album_path].append(str(track_path))
        
        # Convert albums to compositions format
        compositions_list = []
        for album_path, album_tracks in tqdm(albums.items(), desc="Processing albums"):
            if not album_tracks:
                continue
            
            # Process first track from album
            track_path = album_tracks[0]
            track_dir = os.path.dirname(track_path)
            track_base = os.path.splitext(os.path.basename(track_path))[0]

            # Extract group name and track name
            group_name = os.path.basename(os.path.dirname(album_path))
            track_name = track_base
            composition_title = f"{group_name} - {track_name}"
            
            composition = {
                "id": len(compositions_list) + 1,
                "title": composition_title,
                "tracks": []
            }
            
            main_track_path = os.path.join(dataset_path, track_dir, track_base + ".opus")
            vocal1_path = os.path.join(dataset_path, track_dir, track_base + "_vocal1.mp3")
            vocal1_dereverb_path = os.path.join(dataset_path, track_dir, track_base + "_vocal1_dereverb.mp3")
            vocal2_path = os.path.join(dataset_path, track_dir, track_base + "_vocal2.mp3")
            vocal2_dereverb_path = os.path.join(dataset_path, track_dir, track_base + "_vocal2_dereverb.mp3")
            vocal2_dereverb_json_path = os.path.join(dataset_path, track_dir, track_base + "_vocal2_dereverb.json")
            
            if os.path.exists(main_track_path) and os.path.exists(vocal1_path) and os.path.exists(vocal1_dereverb_path) and os.path.exists(vocal2_path) and os.path.exists(vocal2_dereverb_path) and os.path.exists(vocal2_dereverb_json_path):
                composition["tracks"] = [
                    {
                        "title": "Main Track",
                        "type": "main",
                        "url": f"{get_server_url()}/audio/{os.path.relpath(main_track_path, dataset_path)}",
                        "markers": []
                    },
                    {
                        "title": "Vocal 1",
                        "type": "vocal1",
                        "url": f"{get_server_url()}/audio/{os.path.relpath(vocal1_path, dataset_path)}",
                        "markers": []
                    },
                    {
                        "title": "Vocal 2",
                        "type": "vocal2",
                        "url": f"{get_server_url()}/audio/{os.path.relpath(vocal2_path, dataset_path)}",
                        "markers": []
                    },
                    {
                        "title": "Vocal 1 (Dereverb)",
                        "type": "vocal1_dereverb",
                        "url": f"{get_server_url()}/audio/{os.path.relpath(vocal1_dereverb_path, dataset_path)}",
                        "markers": []
                    },
                    {
                        "title": "Vocal 2 (Dereverb)",
                        "type": "vocal2_dereverb",
                        "url": f"{get_server_url()}/audio/{os.path.relpath(vocal2_dereverb_path, dataset_path)}",
                        "markers": []
                    }
                ]

                # add markers from json
                with open(vocal2_dereverb_json_path, 'r') as f:
                    json_data = json.load(f)
                
                # Convert markers format
                converted_markers = []
                segments = []
                if "segments" in json_data:
                    for idx, segment in enumerate(json_data["segments"], 1):
                        # Convert milliseconds to seconds
                        start_sec = segment["start_ms"] / 1000
                        end_sec = segment["end_ms"] / 1000
                        
                        phi4_text = segment.get("phi4_text", "")
                        whisper3_text = segment.get("whisper3_text", "")
                        nemo_text = segment.get("nemo_text", "")
                        
                        # process texts with highlighting
                        phi4_text, whisper3_text, nemo_text = highlight_diffs(phi4_text, whisper3_text, nemo_text)
                        
                        marker = {
                            "start": start_sec,
                            "end": end_sec,
                            "label": f"label_{idx}",
                            "phi4_text": phi4_text,
                            "whisper3_text": whisper3_text,
                            "nemo_text": nemo_text,
                        }
                        
                        converted_markers.append(marker)

                        segment = {
                            "label": f"label_{idx}",
                            "phi4_text": phi4_text,
                            "whisper3_text": whisper3_text,
                            "nemo_text": nemo_text
                        }

                        segments.append(segment)
                
                composition["tracks"][4]["markers"] = converted_markers
                composition["tracks"][4]["segments"] = segments

                compositions_list.append(composition)
        
        compositions = compositions_list
        logger.info(f"Successfully loaded {len(compositions)} compositions")
        logger.info(f"Server running at {get_server_url()}")
        
    except Exception as error:
        logger.error(f'Error initializing dataset: {str(error)}')
        compositions = []

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
    dataset_path = Path("/media/k4_nas/disk1/Datasets/Music_Part1")
    file_path = os.path.join(dataset_path, filename)
    
    if not os.path.exists(file_path):
        logger.error(f"Audio file not found: {file_path}")
        return jsonify({"error": "File not found"}), 404
    
    # Get file size and mime type
    file_size = os.path.getsize(file_path)
    mime_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
    
    # Handle range request
    range_header = request.headers.get('Range')
    
    if range_header:
        try:
            ranges = range_header.replace('bytes=', '').split('-')
            start = int(ranges[0]) if ranges[0] else 0
            end = int(ranges[1]) if ranges[1] else file_size - 1
            
            # Ensure valid range
            if start >= file_size:
                return jsonify({"error": "Invalid range"}), 416
            
            chunk_size = end - start + 1
            
            # Create response with partial content
            response = send_file(
                file_path,
                mimetype=mime_type,
                as_attachment=False,
                conditional=True
            )
            
            response.headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Content-Length'] = str(chunk_size)
            response.status_code = 206
            
            return response
            
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid range request: {str(e)}")
            return jsonify({"error": "Invalid range"}), 416
    
    # No range request, serve entire file
    response = send_file(
        file_path,
        mimetype=mime_type,
        as_attachment=False,
        conditional=True
    )
    
    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['Content-Length'] = str(file_size)
    
    return response
    

@app.route('/api/rescan', methods=['POST'])
def rescan():
    try:
        initialize_dataset()
        return jsonify({
            "message": "Dataset rescanned successfully",
            "count": len(compositions)
        })
    except Exception as error:
        return jsonify({
            "error": f"Failed to rescan dataset: {str(error)}"
        }), 500

if __name__ == '__main__':
    # Only initialize the dataset in the main process when in debug mode
    # When debug=True, Flask spawns a reloader process that would cause the initialization to run twice
    if not os.environ.get('WERKZEUG_RUN_MAIN'):
        print(f"Starting server on http://{HOST}:{PORT}")
        print(f"You can access the server at http://localhost:{PORT} or http://<your-ip-address>:{PORT}")
    else:
        # Initialize only in the main process
        initialize_dataset()
    
    # Start the server with threaded=True for better performance
    app.run(host=HOST, port=PORT, debug=True, threaded=True) 