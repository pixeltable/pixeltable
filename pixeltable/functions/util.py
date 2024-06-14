import PIL.Image


def resolve_torch_device(device: str) -> str:
    import torch
    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return device


def normalize_image_mode(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Converts grayscale images to 3-channel for compatibility with models that only work with
    multichannel input.
    """
    if image.mode == '1' or image.mode == 'L':
        return image.convert('RGB')
    if image.mode == 'LA':
        return image.convert('RGBA')
    return image
