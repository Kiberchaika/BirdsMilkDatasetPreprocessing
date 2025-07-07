#!/usr/bin/env python3
"""
Visualize pitch extraction results over mel-spectrogram using Plotly
"""
import os
import sys
import glob
import torch
import argparse
import numpy as np
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path

def load_pitch_data(pt_path):
    """Load pitch data from .pt file"""
    try:
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
        return data
    except Exception as e:
        print(f"Error loading {pt_path}: {e}")
        return None

def create_mel_spectrogram(audio_path, sr=16000, n_mels=256, hop_length=160):
    """Create mel-spectrogram from audio file"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Create mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=hop_length
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max(mel_spec))
        
        if mel_spec_db is None:
            print(f"mel_spec_db is None for {audio_path}, returning early.")
            return None, None, None, None
        
        # Create time axis
        time_frames = librosa.frames_to_time(
            np.arange(mel_spec_db.shape[1]), sr=sr, hop_length=hop_length
        )
        
        # Create frequency axis
        freq_bins = librosa.mel_frequencies(n_mels=n_mels)
        
        return mel_spec_db, time_frames, freq_bins, sr
        
    except Exception as e:
        print(f"Error creating mel-spectrogram for {audio_path}: {e}")
        return None, None, None, None

def align_pitch_to_mel(pitch_data, mel_time_frames, audio_sr):
    """Align pitch data to mel-spectrogram time frames"""
    if not hasattr(align_pitch_to_mel, "_printed_debug"):
        print(f"[DEBUG] pitch_data content: {pitch_data}")
        align_pitch_to_mel._printed_debug = True
    
    # Robust extraction of pitch values
    if not isinstance(pitch_data, dict) or 'pitch' not in pitch_data:
        print("[WARN] pitch_data missing 'pitch' key or not a dict, skipping.")
        return np.full_like(mel_time_frames, np.nan)
    pitch = pitch_data['pitch']
    # Convert to numpy array and flatten
    pitch_values = np.array(pitch, dtype=float).flatten()
    if pitch_values.size == 0:
        print("[WARN] pitch_values is empty after flattening, skipping.")
        return np.full_like(mel_time_frames, np.nan)
    print(f"[DEBUG] pitch_values type: {type(pitch_values)}, shape: {getattr(pitch_values, 'shape', 'N/A')}")
    print(f"[DEBUG] pitch_values shape after flatten: {pitch_values.shape}")
    
    # Simple stretch/compress pitch values to match mel time frames
    target_length = len(mel_time_frames)
    if len(pitch_values) == 0:
        return np.full(target_length, np.nan)
    
    # Create evenly spaced indices for both source and target
    source_indices = np.linspace(0, len(pitch_values) - 1, len(pitch_values))
    target_indices = np.linspace(0, len(pitch_values) - 1, target_length)
    
    # Simple linear interpolation
    pitch_interp = np.interp(target_indices.tolist(), source_indices.tolist(), pitch_values.tolist())
    
    # Set unvoiced regions to NaN for better visualization
    pitch_interp[pitch_interp <= 0] = np.nan
    
    return pitch_interp

def visualize_pitch(audio_path, crepe_path=None, rmvpe_path=None, ppt_path=None, penn_path=None, output_path=None):
    """Create visualization with mel-spectrogram and pitch overlays"""
    
    print(f"Processing: {os.path.basename(audio_path)}")
    
    # Create mel-spectrogram
    mel_spec_db, time_frames, freq_bins, sr = create_mel_spectrogram(audio_path)
    
    if mel_spec_db is None:
        print("Failed to create mel-spectrogram")
        return
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[f"Pitch Extraction Comparison - {os.path.basename(audio_path)}"]
    )
    
    # Add mel-spectrogram as heatmap
    fig.add_trace(
        go.Heatmap(
            z=mel_spec_db,
            x=time_frames,
            y=freq_bins,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="Magnitude (dB)", x=1.02),
            hovertemplate="Time: %{x:.3f}s<br>Frequency: %{y:.1f}Hz<br>Magnitude: %{z:.1f}dB<extra></extra>",
            name="Mel-Spectrogram"
        )
    )

    # Load and add RMVPE pitch if available
    if rmvpe_path and os.path.exists(rmvpe_path):
        rmvpe_data = load_pitch_data(rmvpe_path)
        if rmvpe_data:
            rmvpe_pitch = align_pitch_to_mel(rmvpe_data, time_frames, sr)
            fig.add_trace(
                go.Scatter(
                    x=time_frames,
                    y=rmvpe_pitch,
                    mode='lines',
                    name='RMVPE Pitch',
                    line=dict(color='blue', width=2),
                    hovertemplate="Time: %{x:.3f}s<br>Pitch: %{y:.1f}Hz<extra></extra>",
                    connectgaps=False
                )
            )
            print(f"Added RMVPE pitch: {os.path.basename(rmvpe_path)}")

    # Load and add CREPE pitch if available
    if crepe_path and os.path.exists(crepe_path):
        crepe_data = load_pitch_data(crepe_path)
        if crepe_data:
            crepe_pitch = align_pitch_to_mel(crepe_data, time_frames, sr)
            fig.add_trace(
                go.Scatter(
                    x=time_frames,
                    y=crepe_pitch,
                    mode='lines',
                    name='CREPE Pitch',
                    line=dict(color='red', width=2),
                    hovertemplate="Time: %{x:.3f}s<br>Pitch: %{y:.1f}Hz<extra></extra>",
                    connectgaps=False
                )
            )
            print(f"Added CREPE pitch: {os.path.basename(crepe_path)}")

    # Load and add PPT pitch if available
    if ppt_path and os.path.exists(ppt_path):
        ppt_data = load_pitch_data(ppt_path)
        if ppt_data:
            ppt_pitch = align_pitch_to_mel(ppt_data, time_frames, sr)
            fig.add_trace(
                go.Scatter(
                    x=time_frames,
                    y=ppt_pitch,
                    mode='lines',
                    name='PPT Pitch',
                    line=dict(color='green', width=2),
                    hovertemplate="Time: %{x:.3f}s<br>Pitch: %{y:.1f}Hz<extra></extra>",
                    connectgaps=False
                )
            )
            print(f"Added PPT pitch: {os.path.basename(ppt_path)}")

    # Load and add PENN pitch if available
    if penn_path and os.path.exists(penn_path):
        penn_data = load_pitch_data(penn_path)
        if penn_data:
            penn_pitch = align_pitch_to_mel(penn_data, time_frames, sr)
            fig.add_trace(
                go.Scatter(
                    x=time_frames,
                    y=penn_pitch,
                    mode='lines',
                    name='PENN Pitch',
                    line=dict(color='purple', width=2),
                    hovertemplate="Time: %{x:.3f}s<br>Pitch: %{y:.1f}Hz<extra></extra>",
                    connectgaps=False
                )
            )
            print(f"Added PENN pitch: {os.path.basename(penn_path)}")
    
    # Update layout
    fig.update_layout(
        title=f"Pitch Extraction Comparison - {os.path.basename(audio_path)}",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=600,
        width=1200,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    # Set y-axis range to focus on pitch frequencies
    fig.update_yaxes(range=[0, 800])
    
    # Save or show
    if output_path:
        fig.write_html(output_path)
        print(f"Saved visualization: {output_path}")
    else:
        fig.show()

def find_matching_files(input_dir):
    """Find matching audio and pitch files"""
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.opus', '*.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    matches = []
    for audio_file in audio_files:
        base_name = os.path.splitext(audio_file)[0]
        crepe_file = f"{base_name}_pitch_crepe.pt"
        rmvpe_file = f"{base_name}_pitch_rmvpe.pt"
        ppt_file = f"{base_name}_pitch_ppt.pt"
        penn_file = f"{base_name}_pitch_penn.pt"
        
        # Check if at least one pitch file exists
        if os.path.exists(crepe_file) or os.path.exists(rmvpe_file) or os.path.exists(ppt_file) or os.path.exists(penn_file):
            matches.append({
                'audio': audio_file,
                'crepe': crepe_file if os.path.exists(crepe_file) else None,
                'rmvpe': rmvpe_file if os.path.exists(rmvpe_file) else None,
                'ppt': ppt_file if os.path.exists(ppt_file) else None,
                'penn': penn_file if os.path.exists(penn_file) else None
            })
    
    return matches

def main():
    parser = argparse.ArgumentParser(description='Visualize pitch extraction results over mel-spectrogram')
    parser.add_argument('-i', '--input', default='./input', type=str,
                       help='Input directory containing audio and .pt files, or single audio file')
    parser.add_argument('-o', '--output', default='./output', type=str,
                       help='Output directory for HTML files (default: same as input)')
    parser.add_argument('--crepe', type=str, default=None,
                       help='Specific CREPE .pt file (for single file mode)')
    parser.add_argument('--rmvpe', type=str, default=None,
                       help='Specific RMVPE .pt file (for single file mode)')
    parser.add_argument('--ppt', type=str, default=None,
                       help='Specific PPT .pt file (for single file mode)')
    parser.add_argument('--penn', type=str, default=None,
                       help='Specific PENN .pt file (for single file mode)')
    parser.add_argument('--show', action='store_true',
                       help='Show interactive plot instead of saving')
    
    args = parser.parse_args()
    
    # Handle single file mode
    if os.path.isfile(args.input):
        base_name = os.path.splitext(args.input)[0]
        crepe_path = args.crepe or f"{base_name}_pitch_crepe.pt"
        rmvpe_path = args.rmvpe or f"{base_name}_pitch_rmvpe.pt"
        ppt_path = args.ppt or f"{base_name}_pitch_ppt.pt"
        penn_path = args.penn or f"{base_name}_pitch_penn.pt"
        
        if args.show:
            output_path = None
        else:
            output_dir = args.output or os.path.dirname(args.input)
            output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(args.input))[0]}_pitch_visualization.html")
        
        visualize_pitch(args.input, crepe_path, rmvpe_path, ppt_path, penn_path, output_path)
        return
    
    # Handle directory mode
    if not os.path.isdir(args.input):
        print(f"Input path does not exist: {args.input}")
        return
    
    matches = find_matching_files(args.input)
    
    if not matches:
        print("No matching audio and pitch files found!")
        return
    
    print(f"Found {len(matches)} files to process")
    
    output_dir = args.output or args.input
    os.makedirs(output_dir, exist_ok=True)
    
    for match in matches:
        try:
            audio_file = match['audio']
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            
            if args.show:
                output_path = None
            else:
                output_path = os.path.join(output_dir, f"{base_name}_pitch_visualization.html")
            
            visualize_pitch(
                audio_file,
                match['crepe'],
                match['rmvpe'],
                match['ppt'],
                match['penn'],
                output_path
            )
            
        except Exception as e:
            print(f"Error processing {match['audio']}: {e}")

if __name__ == "__main__":
    main()