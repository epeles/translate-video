#!/usr/bin/env python3
"""
Script to transcribe YouTube videos using Whisper.
Usage: python transcribe.py "https://youtube.com/watch?v=..." [--model base]
"""

import sys
import os
import re
import tempfile
import argparse
import logging
import whisper
from pytubefix import YouTube
from pytubefix.exceptions import RegexMatchError, VideoUnavailable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_MODELS = ["tiny", "base", "small", "medium", "large"]
YOUTUBE_URL_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w\-]+"
)


def validate_url(url: str) -> None:
    """Raises ValueError if URL doesn't look like a YouTube URL."""
    if not YOUTUBE_URL_PATTERN.match(url):
        raise ValueError(f"Invalid YouTube URL: {url}")


def download_audio(url: str, output_dir: str) -> tuple[str, str]:
    """Downloads audio from a YouTube video."""
    logger.info("Connecting to YouTube...")
    try:
        yt = YouTube(url)
        title = yt.title
    except RegexMatchError:
        raise ValueError(f"Could not parse YouTube URL: {url}")
    except VideoUnavailable as e:
        raise RuntimeError(f"Video unavailable: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to YouTube: {e}")

    logger.info(f"Video: {title}")
    logger.info("Downloading audio...")

    try:
        audio_stream = yt.streams.get_audio_only()
        if audio_stream is None:
            raise RuntimeError("No audio stream available for this video.")
        audio_file = audio_stream.download(output_path=output_dir, filename="audio.mp4")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to download audio: {e}")

    return audio_file, title


def transcribe_audio(audio_path: str, model_name: str = "base") -> str:
    """Transcribes audio using Whisper."""
    logger.info(f"Loading Whisper model '{model_name}'...")
    try:
        model = whisper.load_model(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model '{model_name}': {e}")

    logger.info("Transcribing audio (this may take a while)...")
    try:
        result = model.transcribe(audio_path)
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")

    text = result["text"].strip()
    if not text:
        raise RuntimeError("Transcription produced empty output.")

    return text


def sanitize_filename(name: str) -> str:
    """Removes invalid characters from filename."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")
    return name[:100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a YouTube video using Whisper."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--model",
        default="base",
        choices=VALID_MODELS,
        help="Whisper model to use (default: base). Larger models are more accurate but slower.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        validate_url(args.url)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Processing: {args.url}")
            audio_file, title = download_audio(args.url, temp_dir)

            if not os.path.exists(audio_file):
                raise RuntimeError("Audio file not found after download.")

            transcription = transcribe_audio(audio_file, model_name=args.model)

            output_filename = sanitize_filename(title) + ".txt"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(transcription)

            logger.info(f"Transcription saved to: {output_filename}")

    except (ValueError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
