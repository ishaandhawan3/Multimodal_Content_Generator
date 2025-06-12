# If CogVideoX is not available, use a simpler video generation example
from diffusers import DiffusionPipeline
import torch

class VideoGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Example: Using a lightweight video model if CogVideoX is too large
        self.pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=torch.float16
        ).to(self.device)
    
    def generate(self, prompt, num_frames=8):
        video_frames = self.pipe(prompt, num_frames=num_frames).frames
        return video_frames
