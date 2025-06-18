# Multimodal Content Generator

A project to generate text, images, videos, and audio using open-source LLMs and models.

## Setup

1. Clone this repository.
2. Install dependencies: `./scripts/setup.sh`
3. Copy `.env.example` to `.env` and adjust as needed.
4. Run the app: `streamlit run app/main.py`

## Features

- **Text Generation**: Llama 3.2 Vision
- **Image Generation**: Stable Diffusion 3 Medium
- **Video Generation**: Lightweight text-to-video model
- **Audio Generation**: Bark text-to-speech

## Notes

- For large models, ensure you have enough GPU memory.
- For video and audio, some models may require additional setup or dependencies.
