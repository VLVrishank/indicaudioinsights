
# Indic Audio Insights

**Indic Audio Insights** is an end-to-end speech-to-insight NLP pipeline that processes Hindi YouTube videos and extracts structured information using automatic speech recognition (ASR), POS tagging, named entity recognition (NER), keyword-in-context (KWIC) analysis, and question answering (QA).

## Project Objective

The goal is to demonstrate an applied NLP pipeline for Indian languages using open-source tools. This project showcases the transcription of spoken Hindi audio from YouTube videos and applies natural language processing tasks for further insight extraction.

## Features

- **Audio Downloading**: Extracts audio from a given YouTube URL using `yt_dlp`.
- **Transcription**: Converts speech to text using OpenAI's Whisper ASR model (specifically `medium` for better multilingual performance).
- **Text Cleaning**: Removes unnecessary whitespace and formatting from the transcript.
- **POS Tagging**: Uses `stanza` for part-of-speech tagging in Hindi.
- **Named Entity Recognition (NER)**: Uses `ai4bharat/IndicNER` via HuggingFace Transformers for recognizing named entities in Hindi.
- **KWIC Analysis**: Highlights occurrences of keywords in context within the transcript.
- **Question Answering**: Uses a multilingual transformer (`distilbert-base-multilingual-cased`) to answer questions in Hindi from the transcript.


## Tools and Libraries

- `yt_dlp` - For downloading audio from YouTube.
- `openai-whisper` - For ASR and transcription.
- `stanza` - For POS tagging in Hindi.
- `transformers` (HuggingFace) - For NER (IndicNER) and QA models.
- `torch` - Backend deep learning framework.

## Requirements

Install dependencies with:

```bash
pip install yt-dlp git+https://github.com/openai/whisper.git stanza transformers torch
python -m stanza download hi
