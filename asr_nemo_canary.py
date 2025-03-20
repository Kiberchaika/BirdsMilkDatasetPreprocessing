import os
import librosa
import torch
import numpy as np
from nemo.collections.asr.models import EncDecMultiTaskModel
from typing import Union, Optional

device = torch.device("cuda:1")

# Load the Canary-1B model
print("Loading NVIDIA Canary-1B model...")
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
canary_model = canary_model.to(device)  # Move model to specified GPU

# Configure decoding parameters
decode_cfg = canary_model.cfg.decoding
decode_cfg.beam.beam_size = 5  # Using beam search with width 5
canary_model.change_decoding_strategy(decode_cfg)
canary_model.eval()


def transcribe_using_nemo_canary(source: Union[str, np.ndarray, torch.Tensor], sample_rate: Optional[int] = None) -> str:
    """
    Transcribe audio using NVIDIA's Canary-1B model.
    
    Args:
        source (Union[str, np.ndarray, torch.Tensor]): Either path to the audio file, numpy array, or torch tensor containing audio data
        sample_rate (Optional[int]): Sample rate of the audio data if source is array/tensor. Ignored if source is a file path.
        
    Returns:
        str: The transcribed text
    """
    try:
        # Load and process audio
        if isinstance(source, str):
            print(f"Loading audio file: {source}")
            # Load audio using librosa
            audio, sr = librosa.load(source, sr=None)
            print(f"Successfully loaded audio file. Shape: {audio.shape}, Original Sample rate: {sr}")
        elif isinstance(source, np.ndarray):
            print("Processing provided numpy array")
            audio = source
            sr = sample_rate if sample_rate is not None else 16000
            print(f"Audio array shape: {audio.shape}, Sample rate: {sr}")
        else:  # torch.Tensor
            print("Processing provided torch tensor")
            audio = source.numpy() if source.requires_grad == False else source.detach().numpy()
            sr = sample_rate if sample_rate is not None else 16000
            print(f"Audio tensor shape: {audio.shape}, Sample rate: {sr}")
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            print("Converting stereo to mono...")
            audio = librosa.to_mono(audio)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            print(f"Resampling from {sr}Hz to 16000Hz...")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Transcribe the audio directly
        print("Transcribing audio...")
        
        # Use a simpler approach by using the model's built-in transcribe method with minimal options
        results = canary_model.transcribe(
            audio=[audio],
            batch_size=1,
            num_workers=0,
            task="transcribe",
            source_lang="en",
            target_lang="en",
            pnc="pnc"  # Enable punctuation
        )
        
        return results[0].text
        
        
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    audio_file = '/media/k4_nas/disk1/Datasets/Music/FUNK/10 Speed (2)/10 Speed/10. Yacøpsae - Fratze - Fuck Punk Rock.... This Is Turbo Speed Violence.mp3'
    transcription = transcribe_using_nemo_canary(audio_file)
    print("\n===== TRANSCRIPTION RESULT =====\n")
    print(transcription)
    print("\n================================\n")