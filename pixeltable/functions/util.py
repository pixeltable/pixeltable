def resolve_torch_device(device: str) -> str:
    import torch
    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return device
