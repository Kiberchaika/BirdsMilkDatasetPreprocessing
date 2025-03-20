#!/usr/bin/env python3
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import HfHubHTTPError
import os
import requests
from pathlib import Path
from tqdm import tqdm

# Configuration
REPO_ID = "amphion/Emilia-Dataset"
REPO_TYPE = "dataset"
TARGET_DIR = "/media/k4_nas/disk2/Emilia-Dataset"
MAX_RETRIES = 10
CHUNK_SIZE = 8192

def get_hf_token():
    """Get Hugging Face token from cache"""
    from huggingface_hub import HfFolder
    token = HfFolder.get_token()
    if not token:
        raise ValueError("No Hugging Face token found. Run 'huggingface-cli login' first.")
    return token

def get_file_size(url, headers):
    """Get file size using HEAD request"""
    with requests.head(url, headers=headers, allow_redirects=True) as response:
        response.raise_for_status()
        return int(response.headers.get('content-length', 0))

def download_file(url, file_path, headers, file_size=None):
    """Download a single file with progress bar"""
    if file_size is None:
        file_size = get_file_size(url, headers)
    
    with requests.get(url, headers=headers, stream=True, timeout=30) as response:
        response.raise_for_status()
        
        progress = tqdm(
            total=file_size,
            unit='B',
            unit_scale=True,
            desc=os.path.basename(file_path),
            leave=False
        )
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    progress.update(len(chunk))
        
        progress.close()
        return file_size

def should_download_file(file_path, url, headers):
    """Check if file should be downloaded by comparing sizes"""
    # If file doesn't exist, download it
    if not os.path.exists(file_path):
        return True, None
    
    # Get remote file size
    try:
        remote_size = get_file_size(url, headers)
    except Exception as e:
        tqdm.write(f"Error checking remote file size: {e}")
        return True, None  # If in doubt, download again
    
    # Get local file size
    local_size = os.path.getsize(file_path)
    
    # Compare sizes
    if local_size == remote_size:
        return False, remote_size  # Skip download
    else:
        tqdm.write(f"Size mismatch for {file_path.name}: local={local_size}, remote={remote_size}")
        return True, remote_size  # Re-download

def main():
    # Get authentication token
    token = get_hf_token()
    api = HfApi(token=token)
    
    try:
        # Get file list
        file_list = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    except HfHubHTTPError as e:
        print(f"Access error: {e}")
        print("\nYou might need to:")
        print(f"1. Visit https://huggingface.co/datasets/{REPO_ID}")
        print("2. Click 'Agree and access repository'")
        print("3. Ensure your account has access rights")
        return

    # Create target directory
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Filter out system files
    files_to_download = [
        f for f in file_list
        if not any(f.endswith(ext) for ext in ['.lock', '.gitattributes', '.gitignore'])
    ]

    # Initialize overall progress
    total_files = len(files_to_download)
    downloaded_files = 0
    skipped_files = 0
    total_size = 0  # Will be calculated as we go
    
    with tqdm(
        total=total_files,
        unit='file',
        desc="Overall progress",
        position=0
    ) as pbar_total:
        
        # Download files with proper authentication
        for idx, file in enumerate(files_to_download, 1):
            file_path = Path(TARGET_DIR) / file
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Get download URL
            url = hf_hub_url(
                repo_id=REPO_ID,
                filename=file,
                repo_type=REPO_TYPE
            )

            headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": "Python-requests/2.31.0"
            }

            # Check if we need to download this file
            should_download, remote_size = should_download_file(file_path, url, headers)
            
            if not should_download:
                tqdm.write(f"[{idx}/{total_files}] Skipping {file} (already downloaded)")
                skipped_files += 1
                total_size += remote_size
                pbar_total.update(1)
                continue

            for attempt in range(MAX_RETRIES):
                try:
                    tqdm.write(f"\n[{idx}/{total_files}] Downloading {file}...")
                    file_size = download_file(url, file_path, headers, remote_size)
                    total_size += file_size
                    downloaded_files += 1
                    pbar_total.update(1)
                    break
                except Exception as e:
                    tqdm.write(f"Attempt {attempt+1} failed: {str(e)}")
                    if file_path.exists():
                        file_path.unlink()
            else:
                tqdm.write(f"Permanent failure on {file}")

    print(f"\nDownload complete!")
    print(f"Files downloaded: {downloaded_files}")
    print(f"Files skipped (already downloaded): {skipped_files}")
    print(f"Total processed: {total_size/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()