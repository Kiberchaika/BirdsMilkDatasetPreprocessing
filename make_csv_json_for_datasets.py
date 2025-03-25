import os
import csv
import json
from tqdm import tqdm

# Initialize a set to store unique artist-album pairs
albums_set = set()

# Add genre to mp3 files
need_to_add_genre_to_files = True

# Define the root directories and their required relative path lengths
roots = [
    {
        'path': '/media/k4_nas/disk1/Datasets/Music_RU/Original/',
        'required_length': 2,  # Structure: artist/album, нет жанров
    },
    {
        'path': '/media/k4_nas/disk1/Datasets/Music/',
        'required_length': 3  # Structure: genre/artist/album
    },
    {
        'path': '/media/k4_data/homes/admin/Киберчайка/Датасеты/Megabird_set/Source albums/',
        'required_length': 3  # Structure: genre/artist/album
    },
    {
        'path': '/media/k4_nas/disk1/Datasets/Music_2/10.02',
        'required_length': 3  # Structure: genre/artist/album
    },
    {
        'path': '/media/k4_nas/disk2/Music/04.03',
        'required_length': 3  # Structure: genre/artist/album
    },
    {
        'path': '/media/k4_nas/disk2/Music/07.03',
        'required_length': 3  # Structure: genre/artist/album
    },
    {
        'path': '/media/k4_nas/disk2/Music/17.03',
        'required_length': 3  # Structure: genre/artist/album
    },
    {
        'path': '/media/k4_nas/disk2/Music/27.02',
        'required_length': 3  # Structure: genre/artist/album
    },
]

def count_dirs_at_depth(path, target_depth, current_depth=0):
    """Count number of directories exactly at the target depth"""
    if current_depth > target_depth:
        return 0
    
    if current_depth == target_depth:
        return 1
    
    try:
        count = 0
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir():
                    count += count_dirs_at_depth(entry.path, target_depth, current_depth + 1)
        return count
    except PermissionError:
        return 0

def add_genre_to_files(path, genre):
    with os.scandir(path) as entries:
        for entry in entries:
            base_name = os.path.splitext(entry.name)[0]
            if entry.is_file() and entry.name.lower().endswith('.mp3'): 
                if not any(postfix in base_name.lower() for postfix in ['_vocal', '_instrumental']):
                    mp3_file = entry.name
                    json_path = os.path.join(path, f"{os.path.splitext(mp3_file)[0]}.json")
                    
                    metadata = {'tags': [genre]}
                    
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                existing_metadata = json.load(f)
                                if existing_metadata.get('tags'):
                                    if genre not in existing_metadata['tags']:
                                        existing_metadata['tags'].append(genre)
                                    metadata = existing_metadata
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            # If JSON is invalid or can't be read, we'll create a new one
                            pass
                    
                    # Write the metadata to JSON file
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)

def scan_directory(path, current_depth, required_depth, current_parts=None, pbar=None):
    """Recursively scan directory up to required depth using scandir"""
    if current_parts is None:
        current_parts = []
    
    if current_depth == required_depth:
        if required_depth == 3:
            genre, artist, album = current_parts  
            if need_to_add_genre_to_files:
                add_genre_to_files(path, genre)
        else:
            artist, album = current_parts
        albums_set.add((artist, album))
        if pbar:
            pbar.update(1)
        return
    
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir():
                    new_parts = current_parts + [entry.name]
                    scan_directory(entry.path, current_depth + 1, required_depth, new_parts, pbar)
    except PermissionError:
        pass  # Skip directories we can't access

# Create outer progress bar for root directories
with tqdm(total=len(roots), desc="Processing root directories") as root_pbar:
    for root_info in roots:
        root_path = root_info['path']
        required_length = root_info['required_length']
        
        # Count total directories at the required depth
        total_dirs = count_dirs_at_depth(root_path, required_length - 1)
        
        # Create progress bar for current root's directories
        desc = f"Scanning {os.path.basename(root_path.rstrip('/'))}"
        with tqdm(total=total_dirs, desc=desc, leave=False) as dir_pbar:
            # Process the root directory
            scan_directory(root_path, 0, required_length, None, dir_pbar)
        
        # Update outer progress bar
        root_pbar.update(1)

# Sort results by artist and album
sorted_albums = sorted(albums_set, key=lambda x: (x[0], x[1]))

# Write to CSV
with open('albums_tmp.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Artist', 'Album'])
    for artist, album in sorted_albums:
        writer.writerow([artist, album])

print(f"Successfully exported {len(sorted_albums)} albums to albums.csv")