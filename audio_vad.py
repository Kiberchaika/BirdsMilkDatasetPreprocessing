import librosa
import numpy as np
from pydub import AudioSegment
import os

def convert_to_wav(source_file):
    """Convert any audio file to WAV format."""
    audio = AudioSegment.from_file(source_file)
    wav_path = source_file.rsplit('.', 1)[0] + '_temp.wav'
    audio.export(wav_path, format='wav')
    return wav_path

def detect_segments(source_file, min_silence_len=500, silence_thresh=-40, keep_silence=500):
    """
    Detect non-silent segments in an audio file using librosa's RMSE-based detection.
    
    Parameters:
    - source_file: path to the audio file
    - min_silence_len: minimum length of silence (in ms) to be considered as a split point
    - silence_thresh: silence threshold in dB (relative to the mean RMSE)
    - keep_silence: amount of silence to keep around each segment (in ms)
    
    Returns:
    - List of tuples containing (start_time, end_time) in milliseconds
    """
    try:
        # Load audio file using librosa
        y, sr = librosa.load(source_file, sr=None)
        
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
        
        # Calculate frame length and hop length (in samples)
        frame_length = int(sr * 0.025)  # 25ms frame
        hop_length = int(sr * 0.010)    # 10ms hop
        
        # Calculate RMSE energy for each frame
        rmse = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Convert silence threshold from dB to linear scale relative to mean RMSE
        rmse_mean = np.mean(rmse)
        silence_thresh_linear = rmse_mean * (10 ** (silence_thresh / 20))
        
        # Find silent frames
        silent_frames = rmse < silence_thresh_linear
        
        # Convert frame numbers to time (ms)
        frame_time_ms = (hop_length / sr) * 1000
        
        # Calculate frames for silence parameters
        min_frames = int(min_silence_len / frame_time_ms)
        keep_frames = int(keep_silence / frame_time_ms)
        
        # Find boundaries of non-silent segments
        boundaries = []
        in_segment = False
        current_start = None
        segment_start_idx = None
        last_nonsilent_frame = -1
        
        # Count consecutive silent frames
        silent_count = 0
        
        for i, is_silent in enumerate(silent_frames):
            if not is_silent:
                # Found a non-silent frame
                last_nonsilent_frame = i
                if not in_segment:
                    # Start new segment
                    # Look back to include keep_silence frames before this point
                    segment_start_idx = max(0, i - keep_frames)
                    in_segment = True
                silent_count = 0
            else:
                # Silent frame
                if in_segment:
                    silent_count += 1
                    if silent_count >= min_frames:
                        # End segment
                        # Include keep_silence frames after the last non-silent frame
                        segment_end_idx = min(len(silent_frames), last_nonsilent_frame + keep_frames + 1)
                        boundaries.append((segment_start_idx, segment_end_idx))
                        in_segment = False
                        current_start = None
        
        # Handle the last segment if we're still in one
        if in_segment:
            segment_end_idx = min(len(silent_frames), last_nonsilent_frame + keep_frames + 1)
            boundaries.append((segment_start_idx, segment_end_idx))
        
        # Convert frame numbers to milliseconds
        timestamps = [(int(start * frame_time_ms), int(end * frame_time_ms))
                     for start, end in boundaries]
        
        # Post process timestamps
        processed_timestamps = post_process_timestamps(timestamps)
        
        return processed_timestamps
        
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        return []

def post_process_timestamps(timestamps, min_total_duration_ms=20000, max_gap_ms=1000):
    """
    Post-process the detected timestamps:
    1. Join segments that are close to each other and would form segments
       less than min_total_duration_ms when combined
    
    Parameters:
    - timestamps: List of (start_time, end_time) tuples in milliseconds
    - min_total_duration_ms: Minimum duration in ms for joined segments (default: 20000 ms = 20 sec)
    - max_gap_ms: Maximum gap in ms between segments to consider joining (default: 1000 ms = 1 sec)
    
    Returns:
    - Processed list of (start_time, end_time) tuples
    """
    if not timestamps:
        return []
    
    # Sort timestamps by start time
    sorted_timestamps = sorted(timestamps, key=lambda x: x[0])
    
    processed = []
    current_group = [sorted_timestamps[0]]
    
    for i in range(1, len(sorted_timestamps)):
        current_segment = sorted_timestamps[i]
        last_segment = current_group[-1]
        
        gap = current_segment[0] - last_segment[1]
        
        # Calculate total duration of current group if we add this segment
        start_time = current_group[0][0]
        end_time = max(current_segment[1], current_group[-1][1])
        total_duration = end_time - start_time
        
        # Join if gap is small enough and total duration would still be under threshold
        if gap <= max_gap_ms and total_duration <= min_total_duration_ms:
            current_group.append(current_segment)
        else:
            # Finalize the previous group
            if len(current_group) > 1:
                # Merge the group into a single segment
                group_start = current_group[0][0]
                group_end = max(seg[1] for seg in current_group)
                processed.append((group_start, group_end))
            else:
                # Add the single segment
                processed.append(current_group[0])
            
            # Start a new group
            current_group = [current_segment]
    
    # Handle the last group
    if current_group:
        if len(current_group) > 1:
            # Merge the group into a single segment
            group_start = current_group[0][0]
            group_end = max(seg[1] for seg in current_group)
            processed.append((group_start, group_end))
        else:
            # Add the single segment
            processed.append(current_group[0])
    
    return processed

def extract_segments(source_file, segments, output_dir=None):
    """
    Extract audio segments using librosa.
    
    Parameters:
    - source_file: path to the audio file
    - segments: list of tuples containing (start_time, end_time) in milliseconds
    - output_dir: directory to save the segments (default: same as source file)
    
    Returns:
    - List of paths to the extracted segment files
    """
    try:
        # Load audio file
        y, sr = librosa.load(source_file, sr=None)
        
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
        
        # Prepare output directory
        if output_dir is None:
            output_dir = os.path.dirname(source_file)
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract and save segments
        segment_files = []
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        
        for i, (start_ms, end_ms) in enumerate(segments):
            # Convert milliseconds to samples
            start_sample = int((start_ms / 1000.0) * sr)
            end_sample = int((end_ms / 1000.0) * sr)
            
            # Extract segment
            segment = y[start_sample:end_sample]
            
            # Generate output path
            output_path = os.path.join(output_dir, f"{base_name}_segment_{i+1}.wav")
            
            # Save segment using librosa
            import soundfile as sf
            sf.write(output_path, segment, sr)
            
            segment_files.append(output_path)
        
        return segment_files
        
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        return []

if __name__ == "__main__":
    source_file = '/media/k4_nas/disk1/Datasets/Music/FUNK/10 Speed (2)/10 Speed/10. Yacøpsae - Fratze - Fuck Punk Rock.... This Is Turbo Speed Violence (1)_vocal1_dereverb.mp3'
    source_file = '/media/k4_nas/disk1/Datasets/Music/FUNK/Al Williams (4)/I Am Nothing/21 Al Williams - I Am Nothing (2022_07_04 06_36_36 UTC)_vocal2_dereverb.mp3'
    
    # Detect segments
    print("Detecting segments...")
    segments = detect_segments(source_file)
    #print("Segments detected:", segments)
    print(len(segments))
    # Extract segments
    print("\nExtracting segments...")
    output_files = extract_segments(source_file, segments)
    #print("Segments extracted to:", output_files)


