import os
import json
import librosa
import glob
from audio_vad import detect_segments
from asr_phi4 import transcribe_using_phi4
from asr_whisper3 import transcribe_using_whisper3
from asr_nemo_canary import transcribe_using_nemo_canary
from text_compare import three_way_diff, generate_html
from tqdm import tqdm

"""
json format:

{
    "segments": [
        {
            "start_ms": 1000,
            "end_ms": 2000, 
            "phi4_text": "...",
            "whisper3_text": "...",
            "nemo_text": "..."
        },
        ...
    ]
    "tags": [],
    "language": "en",
}   
""" 

def process_audio_segments(source_file):
    """
    Process audio file by segments and generate transcription comparisons.
    
    Args:
        source_file (str): Path to the source audio file
    """

    # Load the audio file
    print("Loading audio file...")
    audio, sr = librosa.load(source_file, sr=None)
    
    # Detect segments
    print("Detecting segments...")
    segments = detect_segments(source_file)
    print(f"Found {len(segments)} segments")
    
    # Create the output files
    base_name = os.path.splitext(os.path.basename(source_file))[0]
    source_dir = os.path.dirname(source_file)
    output_html = os.path.join(source_dir, f"{base_name}_segments.html")
    output_json = os.path.join(source_dir, f"{base_name}.json")
    
    # Initialize JSON data structure
    json_data = {
        "segments": [],
        "tags": [],
        "language": "en"
    }
    
    # Start HTML content
    html_content = [
        '<!DOCTYPE html>',
        '<html>',
        '<head>',
        '<style>',
        'body { font-family: Arial, sans-serif; margin: 10px; }',
        '.segment { border: 1px solid #ccc; margin: 10px 0; padding: 8px; border-radius: 3px; }',
        '.segment-header { background: #f5f5f5; padding: 10px; margin: -8px -8px 8px -8px; border-radius: 3px 3px 0 0; }',
        '</style>',
        '</head>',
        '<body>',
        f'<h1>Transcription Comparison: {base_name}</h1>'
    ]
    
    # Process each segment with progress bar
    for i, (start_ms, end_ms) in enumerate(tqdm(segments, desc="Processing segments", unit="segment")):
        print(f"\nProcessing segment {i+1}/{len(segments)}")
        
        try:
            # Convert milliseconds to samples
            start_sample = int((start_ms / 1000.0) * sr)
            end_sample = int((end_ms / 1000.0) * sr)
            
            # Extract segment directly from audio
            segment = audio[start_sample:end_sample]
            
            # Get transcriptions from each model
            print("Getting Phi-4 transcription...")
            phi4_text = transcribe_using_phi4(segment, sr)
            
            print("Getting Whisper-3 transcription...")
            whisper3_text = transcribe_using_whisper3(segment, sr)
            
            print("Getting Nemo Canary transcription...")
            nemo_text = transcribe_using_nemo_canary(segment, sr)
            
            # Add segment data to JSON
            segment_data = {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "phi4_text": phi4_text,
                "whisper3_text": whisper3_text,
                "nemo_text": nemo_text
            }
            json_data["segments"].append(segment_data)
            
            # Generate three-way diff and HTML comparison
            marks1, marks2, marks3 = three_way_diff(phi4_text, whisper3_text, nemo_text)
            comparison_html = generate_html(marks1, marks2, marks3, "Phi-4", "Whisper-3", "Nemo Canary")
            
            # Add segment header and wrap comparison in segment div
            segment_html = [
                f'<div class="segment">',
                f'<div class="segment-header">',
                f'<h2>Segment {i+1} ({start_ms/1000:.2f}s - {end_ms/1000:.2f}s)</h2>',
                '</div>',
                comparison_html,
                '</div>'
            ]
            
            # Add segment HTML to main content
            html_content.extend(segment_html)
            
        except Exception as e:
            print(f"Error processing segment {i+1}: {str(e)}")
            html_content.extend([
                f'<div class="segment">',
                f'<div class="segment-header">',
                f'<h3>Segment {i+1} ({start_ms/1000:.2f}s - {end_ms/1000:.2f}s)</h3>',
                '</div>',
                f'<p style="color: red;">Error processing segment: {str(e)}</p>',
                '</div>'
            ])
            continue
    
    # Close HTML content
    html_content.extend([
        '</body>',
        '</html>'
    ])
    
    # Write the complete HTML file
    #with open(output_html, "w", encoding='utf-8') as f:
    #    f.write('\n'.join(html_content))
    
    # Write the JSON file
    with open(output_json, "w", encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nSaved complete comparison to: {output_html}")
    print(f"Saved JSON data to: {output_json}")

if __name__ == "__main__":
    #source_file = '/media/k4_nas/disk1/Datasets/Music/FUNK/10 Speed (2)/10 Speed/10. Yacøpsae - Fratze - Fuck Punk Rock.... This Is Turbo Speed Violence (1)_vocal1_dereverb.mp3'
    #process_audio_segments(source_file)
    
    # Find all matching audio files
    pattern = '/media/k4_nas/disk1/Datasets/Music/FUNK/*/*/*_vocal2_dereverb.mp3'
    audio_files = glob.glob(pattern)
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process each file with progress bar
    for source_file in tqdm(audio_files, desc="Processing files", unit="file"):
        process_audio_segments(source_file)
    
    print("\nCompleted processing all files!")
    