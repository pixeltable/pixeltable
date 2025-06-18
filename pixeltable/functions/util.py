import PIL.Image

from pixeltable.env import Env


def resolve_torch_device(device: str, allow_mps: bool = True) -> str:
    Env.get().require_package('torch')
    import torch

    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if allow_mps and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return device


def normalize_image_mode(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Converts grayscale images to 3-channel for compatibility with models that only work with
    multichannel input.
    """
    if image.mode in ('1', 'L'):
        return image.convert('RGB')
    if image.mode == 'LA':
        return image.convert('RGBA')
    return image
