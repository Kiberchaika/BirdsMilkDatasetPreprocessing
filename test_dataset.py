import torch
from torch.utils.data import Dataset, DataLoader
import time
import os
import threading

class SlowDataset(Dataset):
    """Dataset that simulates slow data loading with a 2-second delay."""
    def __init__(self, size=100, delay=2.0):
        self.size = size
        self.delay = delay
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Print worker information for debugging
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else "main"
        thread_id = threading.get_native_id()
        
        print(f"[{time.strftime('%H:%M:%S')}] Worker {worker_id} (Thread {thread_id}) loading sample {idx}...")
        
        # Simulate slow data loading
        time.sleep(self.delay)
        
        # Generate a simple tensor as our data
        data = torch.tensor([idx], dtype=torch.float32)
        
        print(f"[{time.strftime('%H:%M:%S')}] Worker {worker_id} (Thread {thread_id}) finished loading sample {idx}")
        
        return data
    
def test_dataloader(batch_size, num_workers, prefetch_factor, num_batches=3):
    """Test the dataloader with different parameters and show prefetch behavior."""
    print(f"\nTesting DataLoader with: batch_size={batch_size}, num_workers={num_workers}, prefetch_factor={prefetch_factor}")
    print("=" * 80)
    
    # Create dataset and dataloader
    dataset = SlowDataset(size=100, delay=2.0)
    
    # Configure DataLoader
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'shuffle': False  # Keep it simple and deterministic
    }
    
    # Add prefetch_factor only when num_workers > 0
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
    
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    
    # Process some batches and time it
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Starting to fetch batches...")
    
    for i, batch in enumerate(dataloader):
        batch_receive_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Received batch {i} with shape {batch.shape} after {batch_receive_time - start_time:.2f}s")
        
        # Simulate some processing
        time.sleep(20.0)

        print(batch)
        
        if i >= num_batches - 1:
            break
    
    total_time = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}] Completed {num_batches} batches in {total_time:.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    test_dataloader(batch_size=4, num_workers=2, prefetch_factor=2)
    