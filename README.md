# Birds Milk Dataset Preprocessing

A comprehensive toolkit for preprocessing audio datasets.

## Features

- Audio format conversion (MP3 to Opus)
- Vocal separation and dereverberation from music tracks
- Integration with the Blackbird Dataset

## Project Structure

- `audio_separator.py` - Core functionality for audio separation and processing
- `task_convert_opus.py` - Handles audio format conversion to Opus
- `task_separate.py` and `task_separate2.py` - Audio separation task implementations
- `input/` - Directory for input audio files
- `ckpts/` - Directory for model checkpoints
- `The_Blackbird_Dataset/` - Submodule containing the Blackbird dataset
- `Music-Source-Separation-Training/` - Submodule for music source separation models

## Usage

### Converting Audio to Opus Format

```python
python task_convert_opus.py
```

This script will:
- Process audio files from the Blackbird dataset
- Convert them to Opus format with optimized settings
- Preserve metadata during conversion

### Separating Vocals

```python
python task_separate.py
```

Features:
- Extracts vocals and dereverb from music tracks

## Links

- [The Blackbird Dataset](https://github.com/Kiberchaika/The_Blackbird_Dataset)
- [Music Source Separation Training](https://github.com/jarredou/Music-Source-Separation-Training) 