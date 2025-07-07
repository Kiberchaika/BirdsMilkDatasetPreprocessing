import os
import time
import platform
import subprocess
import tempfile
import shutil
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Create a temporary test directory structure
def create_test_directory_structure(base_dir, num_dirs=50, files_per_dir=20, max_depth=3, current_depth=0):
    """Create a test directory structure with files for benchmarking"""
    os.makedirs(base_dir, exist_ok=True)
    
    # Create some files in this directory
    for i in range(files_per_dir):
        file_path = os.path.join(base_dir, f"file_{i}.txt")
        with open(file_path, 'w') as f:
            f.write(f"This is test file {i}")
    
    # Create subdirectories if we haven't reached max depth
    if current_depth < max_depth:
        num_subdirs = max(2, num_dirs // (current_depth + 1))  # Fewer subdirs as we go deeper
        for j in range(num_subdirs):
            subdir = os.path.join(base_dir, f"subdir_{j}")
            create_test_directory_structure(
                subdir, 
                num_dirs=num_dirs // 2,  # Fewer dirs as we go deeper
                files_per_dir=files_per_dir,
                max_depth=max_depth,
                current_depth=current_depth + 1
            )

# Method 1: Traditional os.walk
def count_files_walk(directories):
    """Count files using os.walk"""
    total_files = 0
    for directory in directories:
        if not os.path.exists(directory):
            continue
        for root, _, files in os.walk(directory):
            total_files += len(files)
    return total_files

# Method 2: Using find command (Unix/Linux/macOS only)
def count_files_find_command(directories):
    """Count files using find command"""
    total_files = 0
    for directory in directories:
        if not os.path.exists(directory):
            continue
        try:
            # Use wc -l to count lines from find output
            cmd = f"find {directory} -type f | wc -l"
            result = subprocess.check_output(cmd, shell=True, text=True)
            count = int(result.strip())
            total_files += count
        except (subprocess.SubprocessError, ValueError) as e:
            print(f"Error using find command: {e}")
            # Fall back to os.walk
            for root, _, files in os.walk(directory):
                total_files += len(files)
    return total_files

# Method 3: Using scandir
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

# Method 4: Using pathlib
def count_files_pathlib(directories):
    """Count files using pathlib"""
    total_files = 0
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
        # Using pathlib to get all files
        path = Path(directory)
        count = sum(1 for _ in path.rglob('*') if _.is_file())
        total_files += count
    
    return total_files

# Method 5: Using parallel processing
def count_files_parallel(directories, max_workers=4):
    """Count files using parallel processing"""
    total_files = 0
    
    def count_in_subdir(path):
        count = 0
        for root, _, files in os.walk(path):
            count += len(files)
        return count
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
        
        # Get first-level subdirectories
        try:
            subdirs = [os.path.join(directory, d) for d in os.listdir(directory) 
                      if os.path.isdir(os.path.join(directory, d))]
            
            # If no subdirs, just count files in the main directory
            if not subdirs:
                for _, _, files in os.walk(directory):
                    total_files += len(files)
                continue
                
            # Use parallel processing on subdirectories
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                counts = list(executor.map(count_in_subdir, subdirs))
                total_files += sum(counts)
                
        except PermissionError:
            pass
    
    return total_files

# Method 6: Estimating file count
def estimate_files_for_progress(directories, sample_size=0.2):
    """Estimate files using sampling"""
    total_estimate = 0
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
        
        # Get all subdirectories
        all_subdirs = []
        for root, dirs, files in os.walk(directory):
            all_subdirs.append(root)
            # Count files in the top level
            if root == directory:
                total_estimate += len(files)
            # Break after getting enough subdirs for sampling
            if len(all_subdirs) > 50:  # Limit to avoid scanning everything
                break
        
        # Sample a percentage of subdirectories
        if len(all_subdirs) > 1:
            sample_count = max(1, int(len(all_subdirs) * sample_size))
            sample_dirs = random.sample(all_subdirs[1:], min(sample_count, len(all_subdirs)-1))
            
            # Count files in sampled subdirectories
            sample_total = 0
            for subdir in sample_dirs:
                for _, _, files in os.walk(subdir):
                    sample_total += len(files)
            
            # Extrapolate to estimate total
            if sample_dirs:
                avg_files_per_dir = sample_total / len(sample_dirs)
                estimated_subdirs = len(all_subdirs) - 1  # -1 for the root dir
                total_estimate += int(avg_files_per_dir * estimated_subdirs)
    
    return int(total_estimate)

# Benchmark function
def benchmark_methods(directories):
    """Benchmark all file counting methods"""
    methods = [
        ("Traditional os.walk", count_files_walk),
        ("os.scandir", count_files_scandir),
        ("pathlib", count_files_pathlib),
        ("Parallel processing", count_files_parallel),
        ("Sampling estimation", estimate_files_for_progress)
    ]
    
    # Add find command method for Unix-like systems
    if platform.system() in ('Linux', 'Darwin'):
        methods.insert(1, ("find command", count_files_find_command))
    
    results = []
    for name, method in methods:
        start_time = time.time()
        try:
            count = method(directories)
            elapsed_time = time.time() - start_time
            results.append((name, count, elapsed_time))
            print(f"{name}: Found {count} files in {elapsed_time:.4f} seconds")
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    return results

def main():
    # Create a test directory structure
    try:
        temp_dir = tempfile.mkdtemp()
        test_dir = os.path.join(temp_dir, "test_structure")
        
        print("Creating test directory structure...")
        create_test_directory_structure(test_dir, num_dirs=15, files_per_dir=10, max_depth=3)
        
        # Count actual number of files for verification
        actual_count = 0
        for root, _, files in os.walk(test_dir):
            actual_count += len(files)
        print(f"Created test structure with {actual_count} files.")
        
        # Run benchmark
        print("\nBenchmarking file counting methods...")
        results = benchmark_methods([test_dir])
        
        # Print summary and comparison
        print("\nResults Summary:")
        print("-" * 60)
        print(f"{'Method':<25} {'Count':<10} {'Time (s)':<10} {'Accuracy':<10}")
        print("-" * 60)
        
        # Sort by speed
        results.sort(key=lambda x: x[2])
        
        for name, count, elapsed_time in results:
            accuracy = (count / actual_count * 100) if actual_count > 0 else 0
            print(f"{name:<25} {count:<10} {elapsed_time:<10.4f} {accuracy:<10.2f}%")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up temporary directory
        if 'temp_dir' in locals():
            try:
                shutil.rmtree(temp_dir)
                print(f"\nCleaned up temporary test directory: {temp_dir}")
            except Exception as e:
                print(f"Error cleaning up: {e}")

if __name__ == "__main__":
    main()