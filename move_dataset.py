#!/usr/bin/env python3
import os
import shutil

# Destination directory
DEST = "/media/k4_nas/disk1/Datasets/Music_Part1"

# Source directories
SOURCES = [
    "/media/k4_nas/disk1/Datasets/Music_RU/Original",
    "/media/k4_nas/disk1/Datasets/Music/NEW WAVE",
    "/media/k4_nas/disk1/Datasets/Music/FUNK",
    "/media/k4_nas/disk1/Datasets/Music/POP",
    "/media/k4_nas/disk1/Datasets/Music/SOUL",
    "/media/k4_nas/disk1/Datasets/Music/JAZZ",
    "/media/k4_nas/disk1/Datasets/Music/RNB",
    "/media/k4_data/homes/admin/Киберчайка/Датасеты/Megabird_set/Source albums/HARD ROCK",
    "/media/k4_data/homes/admin/Киберчайка/Датасеты/Megabird_set/Source albums/new wave",
    "/media/k4_data/homes/admin/Киберчайка/Датасеты/Megabird_set/Source albums/progressive",
    "/media/k4_data/homes/admin/Киберчайка/Датасеты/Megabird_set/Source albums/Инди",
    "/media/k4_nas/disk1/Datasets/Music_2/10.02/Hip_Hop",
    "/media/k4_nas/disk1/Datasets/Music_2/10.02/Latin",
    "/media/k4_nas/disk1/Datasets/Music_2/10.02/Pop",
    "/media/k4_nas/disk1/Datasets/Music_2/10.02/Reggae",
]

'''
# Destination directory
DEST = "/media/k4_nas/disk2/Music_Part2"

# Source directories
SOURCES = [
    "/media/k4_nas/disk2/Music/04.03/Brass_Military",
    "/media/k4_nas/disk2/Music/04.03/Children",
    "/media/k4_nas/disk2/Music/07.03/Stage_Screen",
    "/media/k4_nas/disk2/Music/17.03/Funk_Soul",
    "/media/k4_nas/disk2/Music/27.02/Blues"
]
'''


# Create destination directory if it doesn't exist
os.makedirs(DEST, exist_ok=True)

def should_move_file(src_path, dest_path):
    """Check if file should be moved (either doesn't exist in destination or sizes differ)"""
    if not os.path.exists(dest_path):
        return True
    return False#os.path.getsize(src_path) != os.path.getsize(dest_path)

# Move files from each source directory to the destination
for SRC in SOURCES:
    if os.path.isdir(SRC):
        print(f"Processing files from: {SRC}")
        
        for root, dirs, files in os.walk(SRC):
            for file in files:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, SRC)
                dest_path = os.path.join(DEST, rel_path)
                
                # Create destination directory structure if needed
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                if should_move_file(src_path, dest_path): 
                    try:
                        shutil.move(src_path, dest_path)
                        print(f"Moved: {src_path} -> {dest_path}")
                    except Exception as e:
                        print(f"Error moving {src_path}: {e}")
                else:
                    print(f"Skipping (same size exists): {src_path}")
                    try:
                        os.remove(src_path)  # Remove source file since identical exists in destination
                    except Exception as e:
                        print(f"Error removing {src_path}: {e}")
        
        # Remove empty directories after moving
        for root, dirs, files in os.walk(SRC, topdown=False):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if not os.listdir(dir_path):  # Check if directory is empty
                    try:
                        os.rmdir(dir_path)
                    except Exception as e:
                        print(f"Error removing directory {dir_path}: {e}")
                    
        # Check if the source directory itself is now empty and remove it
        if os.path.exists(SRC) and not os.listdir(SRC):
            try:
                os.rmdir(SRC)
            except Exception as e:
                print(f"Error removing source directory {SRC}: {e}")
    else:
        print(f"Warning: Directory not found -> {SRC}")

print("Move completed.")