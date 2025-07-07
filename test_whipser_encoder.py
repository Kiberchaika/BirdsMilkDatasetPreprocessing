import torch
import torch.nn.functional as F
import numpy as np
import whisper

class WhisperAudioEncoder:
    """
    A class for extracting audio features using only the first two
    convolutional layers of Whisper's encoder.
    """
    
    def __init__(self, model_name="base"):
        """
        Initialize the encoder with a Whisper model.
        
        Parameters:
        -----------
        model_name : str, optional
            Name of the Whisper model to use, default is "base"
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name).to(self.device)
    
    def encode(self, audio_path):
        """
        Extract audio features from an audio file.
        
        Parameters:
        -----------
        audio_path : str
            Path to the audio file
        
        Returns:
        --------
        features : torch.Tensor
            Features extracted from the first two conv layers
        """
        audio = whisper.load_audio(audio_path)
        #audio = whisper.audio.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels).to(self.device)
        print(mel.shape)
        
        with torch.no_grad():
            x = F.gelu(self.model.encoder.conv1(mel.unsqueeze(0)))
            features = F.gelu(self.model.encoder.conv2(x))
            print(features.shape)
        

        with torch.no_grad():
            # Extract features using the encoder
            features = self.model.encoder(mel.unsqueeze(0))
            print(features.shape)


        return features

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="/home/k4/Projects/BirdsMilkDatasetPreprocessing/input/Alkonost Русалка New song_vocal_dereverb.mp3")
    parser.add_argument("--model", type=str, default="large-v3")
    parser.add_argument("--output", type=str, default="audio_features_conv.npy")
    args = parser.parse_args()
    
    encoder = WhisperAudioEncoder(args.model)
    features = encoder.encode(args.audio)
    print(f"Feature shape: {features.shape}")
    
    if args.output:
        np.save(args.output, features.cpu().numpy())