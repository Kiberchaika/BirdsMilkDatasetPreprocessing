import os
import torch
import librosa
import soundfile as sf
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from typing import Union, Optional

device = torch.device("cuda:1")

print("Loading Whisper model and processor...")
model_name = "openai/whisper-large-v3"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    trust_remote_code=True
).to(device)

def transcribe_using_whisper3(source: Union[str, np.ndarray], sample_rate: Optional[int] = None) -> str:
    """
    Transcribe audio using Whisper v3 model.
    
    Args:
        source (Union[str, np.ndarray]): Either path to the audio file or numpy array containing audio data
        sample_rate (Optional[int]): Sample rate of the audio data if source is numpy array. Ignored if source is a file path.
        
    Returns:
        str: Transcribed text from the audio
    """
    # Load and process audio
    if isinstance(source, str):
        print(f"Loading audio file: {source}")
        audio, sr = librosa.load(source, sr=16000)
        print(f"Successfully loaded audio file. Shape: {audio.shape}, Sample rate: {sr}")
    else:
        print("Processing provided audio array")
        audio = source
        sr = sample_rate if sample_rate is not None else 16000
        if sr != 16000:
            print(f"Resampling audio from {sr}Hz to 16000Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        print(f"Audio array shape: {audio.shape}, Sample rate: {sr}")
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        print("Converting stereo to mono...")
        audio = librosa.to_mono(audio)
    
    # Process audio through the model
    print("Processing audio through Whisper...")
    inputs = processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt",
        return_attention_mask=True  # Explicitly request attention mask
    )
    
    # Convert input features to same dtype as model
    input_features = inputs.input_features.to(device=device, dtype=model.dtype)
    attention_mask = inputs.attention_mask.to(device=device)
    
    # Generate transcription
    print("Generating transcription...")
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            max_length=448,
            num_beams=10,  # Increased from 5
            temperature=0.2,  # Added some temperature for better handling of unclear audio
            language="en",  # Force English
            task="transcribe",  # Force transcription
            no_repeat_ngram_size=3,  # Prevent repetition
            num_return_sequences=1,
            do_sample=True,  # Enable sampling
            top_k=50,  # Restrict to top 50 tokens
            top_p=0.95  # Nucleus sampling
        )
    
    # Decode and return transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()
 
if __name__ == "__main__":
    source_file = '/media/k4_nas/disk1/Datasets/Music/FUNK/10 Speed (2)/10 Speed/10. Yacøpsae - Fratze - Fuck Punk Rock.... This Is Turbo Speed Violence.mp3'
    result = transcribe_using_whisper3(source_file)
    print("\n===== TRANSCRIPTION RESULT =====\n")
    print(result)
    print("\n================================\n")
