import os

class Config:
    LLAMA_MODEL_SIZE = os.getenv("LLAMA_MODEL_SIZE", "11B")
    FLUX_MODEL_VARIANT = os.getenv("FLUX_MODEL_VARIANT", "schnell")
    COGVIDEO_MODEL_SIZE = os.getenv("COGVIDEO_MODEL_SIZE", "5B")
    AUDIO_MODEL_TYPE = os.getenv("AUDIO_MODEL_TYPE", "stable-audio-open")
