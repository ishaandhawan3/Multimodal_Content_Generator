from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class TextGenerator:
    def __init__(self, model_size="11B"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_map = {
            "11B": "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "90B": "meta-llama/Llama-3.2-90B-Vision-Instruct"
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_map[model_size])
        self.model = AutoModelForCausalLM.from_pretrained(
            model_map[model_size],
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def generate(self, prompt, max_length=512):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
