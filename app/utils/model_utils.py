def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_memory_usage():
    import torch
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
