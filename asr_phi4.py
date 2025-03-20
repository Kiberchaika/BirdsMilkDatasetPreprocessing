import os
import torch
import numpy as np
import librosa
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from typing import Union, Optional

device = torch.device("cuda:0")

# Initialize global model, processor and generation config
print("Loading processor from microsoft/Phi-4-multimodal-instruct...")
processor = AutoProcessor.from_pretrained("microsoft/Phi-4-multimodal-instruct", trust_remote_code=True)

print("Loading model from microsoft/Phi-4-multimodal-instruct...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-multimodal-instruct",
    trust_remote_code=True,
    torch_dtype='auto',
    attn_implementation='flash_attention_2',
    device_map=device,
)

# Load generation config
generation_config = GenerationConfig.from_pretrained("microsoft/Phi-4-multimodal-instruct", 'generation_config.json')

print("Model configuration:")
print("Attention implementation:", model.config._attn_implementation)
print(generation_config)


def transcribe_using_phi4(source: Union[str, np.ndarray], sample_rate: Optional[int] = None) -> str:
    """
    Transcribe audio using Microsoft's Phi-4-multimodal-instruct model.
    
    Args:
        source (Union[str, np.ndarray]): Either path to the audio file or numpy array containing audio data
        sample_rate (Optional[int]): Sample rate of the audio data if source is numpy array. Ignored if source is a file path.
        
    Returns:
        str: The transcription text
    """
    try:
        # Load and process audio
        if isinstance(source, str):
            print(f"Reading audio file: {source}")
            audio_array, sr = librosa.load(source, sr=44100)
        else:
            print("Processing provided audio array")
            audio_array = source
            sr = sample_rate if sample_rate is not None else 44100
            if sr != 44100:
                print(f"Resampling audio from {sr}Hz to 44100Hz")
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=44100)
        
        audio = [audio_array, 44100]
        
        # Create prompt with chat template
        speech_prompt = "Based on the attached audio, generate a comprehensive text transcription of the spoken content."
        chat = [{'role': 'user', 'content': f'<|audio_1|>{speech_prompt}'}]
        
        # Use the chat template from the tokenizer
        prompt = processor.tokenizer.apply_chat_template(
            chat, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Remove endoftext token if present
        if prompt.endswith('<|endoftext|>'):
            prompt = prompt.rstrip('<|endoftext|>')
            
        print(f'>>> Prompt\n{prompt}')
        
        print("Processing audio...")
        inputs = processor(
            text=prompt,
            audios=[audio],
            return_tensors='pt'
        ).to(device)
        
        print("Generating transcription...")
        # Generation arguments
        generation_args = {
            'max_new_tokens': 1000,
            'num_logits_to_keep': 0,
            'temperature': 0.0,
            'do_sample': False,
        }
        
        generate_ids = model.generate(
            **inputs,
            **generation_args,
            generation_config=generation_config,
        )
        
        # Extract only the generated tokens
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        
        # Decode the generated tokens to text
        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip()
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise


if __name__ == "__main__":
    # Replace with the path to your audio file
    audio_file = '/media/k4_nas/disk1/Datasets/Music/FUNK/10 Speed (2)/10 Speed/10. Yacøpsae - Fratze - Fuck Punk Rock.... This Is Turbo Speed Violence.mp3'
    
    print("Starting transcription process...")
    result = transcribe_using_phi4(audio_file)
    
    print("\n===== TRANSCRIPTION RESULT =====\n")
    print(result)
    print("\n================================\n")