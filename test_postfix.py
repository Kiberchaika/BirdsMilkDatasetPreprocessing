import os
import re
import json
from collections import defaultdict
from tqdm import tqdm


def count_files_scandir(directories):
    """Count files using os.scandir"""
    total_files = 0
    
    def count_in_dir(path):
        count = 0
        try:
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_file():
                        count += 1
                    elif entry.is_dir():
                        count += count_in_dir(entry.path)
        except PermissionError:
            pass
        return count
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
        total_files += count_in_dir(directory)
    
    return total_files

def scan_directories(directories):
    """
    Scan all files in the given directories and extract extensions and postfix parts.
    
    Args:
        directories (list): List of directory paths to scan
        
    Returns:
        tuple: (unique_extensions, postfix_dict)
    """
    unique_extensions = set()
    postfix_dict = defaultdict(int)
    
    # Regular expression to match postfix parts: FILENAME_[postfix_with_extension]
    # This pattern looks for an underscore followed by any characters until the end of the filename
    postfix_pattern = re.compile(r'_([^_/\\\\]+)$')
    
    # Count total files first for progress bar
    total_files = count_files_scandir(directories)
    
    print(f"Found {total_files} files to process")
    
    # Create a progress bar for all files
    progress_bar = tqdm(total=total_files, desc="Processing files", unit="file")
    
    def scan_dir_recursive(path):
        try:
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_file():
                        progress_bar.update(1)
                        filename = entry.name
                        # Get file extension
                        _, ext = os.path.splitext(filename)
                        if ext:
                            # Remove the dot and convert to lowercase
                            ext = ext[1:].lower()
                            unique_extensions.add(ext)

                        # Find postfix parts
                        match = postfix_pattern.search(filename)
                        if match:
                            postfix = match.group(1)
                            postfix_dict[postfix] += 1
                    elif entry.is_dir():
                        scan_dir_recursive(entry.path)
        except PermissionError:
            # Handle permission errors if necessary
            print(f"Permission denied: {path}")
            pass
        except FileNotFoundError:
            print(f"Directory not found: {path}")
            pass

    for directory in directories:
        if not os.path.exists(directory):
            continue

        print(f"Scanning directory: {directory}")
        scan_dir_recursive(directory)

    # Ensure the progress bar closes properly even if total count was slightly off
    progress_bar.close()

    return unique_extensions, postfix_dict

def save_results(unique_extensions, postfix_dict, output_dir="./results"):
    """
    Save the unique extensions and postfix parts to files.
    
    Args:
        unique_extensions (set): Set of unique file extensions
        postfix_dict (dict): Dictionary of postfix parts and their counts
        output_dir (str): Directory to save the results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save unique extensions
    extensions_file = os.path.join(output_dir, "unique_extensions.txt")
    with open(extensions_file, 'w') as f:
        for ext in sorted(unique_extensions):
            f.write(f"{ext}\\n")
    
    # Save postfix parts with counts
    postfix_file = os.path.join(output_dir, "postfix_parts.json")
    with open(postfix_file, 'w') as f:
        json.dump(postfix_dict, f, indent=4, sort_keys=True)
    
    # Also save postfix parts sorted by frequency (most common first)
    sorted_postfix_file = os.path.join(output_dir, "postfix_parts_by_frequency.txt")
    with open(sorted_postfix_file, 'w') as f:
        for postfix, count in sorted(postfix_dict.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{postfix}: {count}\\n")
    
    print(f"Results saved to {output_dir}:")
    print(f"  - {len(unique_extensions)} unique extensions saved to {extensions_file}")
    print(f"  - {len(postfix_dict)} postfix patterns saved to {postfix_file} and {sorted_postfix_file}")
    print(f"  - Total files with postfix patterns: {sum(postfix_dict.values())}")

def main():
    # Directories to scan
    directories = [
        "/media/k4_nas/disk1/Datasets/Music_Part1",
        "/media/k4_nas/disk2/Music_Part2"
    ]
    
    # Scan directories
    print("Starting scan of music directories...")
    unique_extensions, postfix_dict = scan_directories(directories)
    
    # Save results
    save_results(unique_extensions, postfix_dict)
    
    
    
    # Print summary
    print("\\nScan complete!")
    print(f"Found {len(unique_extensions)} unique file extensions")
    print(f"Found {len(postfix_dict)} unique postfix patterns in {sum(postfix_dict.values())} files")
    
    # Print some examples of the most common postfix patterns
    print("\\nMost common postfix patterns:")
    for postfix, count in sorted(postfix_dict.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {postfix}: {count} occurrences")

if __name__ == "__main__":
    main()