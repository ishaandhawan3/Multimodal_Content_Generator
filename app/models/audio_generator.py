from transformers import AutoProcessor, BarkModel
import torch

class AudioGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        self.model = BarkModel.from_pretrained("suno/bark").to(self.device)
    
    def generate_speech(self, text, voice_preset="v2/en_speaker_6"):
        inputs = self.processor(text, voice_preset=voice_preset)
        return self.model.generate(**inputs.to(self.device))
