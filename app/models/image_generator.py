from diffusers import StableDiffusion3Pipeline
import torch

class ImageGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16
        ).to(self.device)
    
    def generate(self, prompt, negative_prompt=""):
        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=7.0,
            num_inference_steps=25
        ).images[0]
