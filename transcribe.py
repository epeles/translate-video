#!/usr/bin/env python3
"""
Script to transcribe YouTube videos using Whisper.
Usage: python transcribe.py "https://youtube.com/watch?v=..."
"""

import sys
import os
import tempfile
import whisper
from pytubefix import YouTube


def download_audio(url: str, output_dir: str) -> tuple[str, str]:
    """Downloads audio from a YouTube video."""
    print(f"Connecting to YouTube...")
    yt = YouTube(url)
    title = yt.title
    
    print(f"Video: {title}")
    print("Downloading audio...")
    
    audio_stream = yt.streams.get_audio_only()
    audio_file = audio_stream.download(output_path=output_dir, filename="audio.mp4")
    
    return audio_file, title


def transcribe_audio(audio_path: str, model_name: str = "base") -> str:
    """Transcribes audio using Whisper."""
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    
    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    
    return result["text"]


def sanitize_filename(name: str) -> str:
    """Removes invalid characters from filename."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name[:100]


def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <YOUTUBE_URL>")
        sys.exit(1)
    
    url = sys.argv[1]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Downloading audio from: {url}")
        audio_file, title = download_audio(url, temp_dir)
        
        if not os.path.exists(audio_file):
            print("Error: Failed to download audio.")
            sys.exit(1)
        
        transcription = transcribe_audio(audio_file)
        
        output_filename = sanitize_filename(title) + ".txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(transcription)
        
        print(f"\nTranscription saved to: {output_filename}")


if __name__ == "__main__":
    main()
