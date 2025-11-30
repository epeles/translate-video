# YouTube Video Transcription

Python script to transcribe YouTube videos using OpenAI Whisper.

## Features

- Downloads audio from YouTube videos
- Transcribes audio using Whisper (local model)
- Saves transcription to a text file

## Requirements

- Python 3.9+
- ffmpeg (install with `brew install ffmpeg` on macOS)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python transcribe.py "https://youtube.com/watch?v=..."
```

## Dependencies

- `openai-whisper` - Speech recognition model
- `pytubefix` - YouTube video downloader

## Notes

- The script uses the Whisper "base" model by default. You can modify the model in the code for better accuracy (e.g., "small", "medium", "large").
- The first run will download the Whisper model (~139MB for base model).
- Transcription quality depends on audio quality and language.

