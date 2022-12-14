import contextlib
import copy
import gc
import random
from functools import wraps
from typing import Dict, List, Tuple, TypeVar

import numpy as np
import ray
import torch
from PIL import Image

T = TypeVar("T")


def seed_everything(seed: int = None, workers: bool = False) -> int:
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    def select_seed_randomly(min_seed_value: int = min_seed_value, max_seed_value: int = max_seed_value) -> int:
        return random.randint(min_seed_value, max_seed_value)

    if not seed or not (min_seed_value <= seed <= max_seed_value):
        seed = select_seed_randomly()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def performance(f: T) -> T:
    @wraps(f)
    def wrapper(*args, **kwargs):
        return torch.cuda.amp.autocast()(torch.no_grad()(f))(*args, **kwargs)

    return wrapper


def extract_tensors(m: torch.nn.Module) -> Tuple[torch.nn.Module, List[Dict]]:
    tensors = []
    for _, module in m.named_modules():
        params = {
            name: torch.clone(param).cpu().detach().numpy() for name, param in module.named_parameters(recurse=False)
        }
        buffers = {name: torch.clone(buf).cpu().detach().numpy() for name, buf in module.named_buffers(recurse=False)}
        tensors.append({"params": params, "buffers": buffers})

    m_copy = copy.deepcopy(m)
    for _, module in m_copy.named_modules():
        for name in [name for name, _ in module.named_parameters(recurse=False)] + [
            name for name, _ in module.named_buffers(recurse=False)
        ]:
            setattr(module, name, None)

    m_copy.train(False)
    return m_copy, tensors


def replace_tensors(m: torch.nn.Module, tensors: List[Dict], device="cuda"):
    modules = [module for _, module in m.named_modules()]
    for module, tensor_dict in zip(modules, tensors):
        # There are separate APIs to set parameters and buffers.
        for name, array in tensor_dict["params"].items():
            module.register_parameter(
                name,
                torch.nn.Parameter(torch.as_tensor(array, device=device), requires_grad=False),
            )
        for name, array in tensor_dict["buffers"].items():
            module.register_buffer(name, torch.as_tensor(array, device=device))


@contextlib.contextmanager
def load_from_plasma(ref, device="cuda"):
    skeleton, weights = ray.get(ref)
    replace_tensors(skeleton, weights, device=device)
    skeleton.eval().half().to(device)
    yield skeleton
    torch.cuda.empty_cache()


def push_model_to_plasma(model: torch.nn.Module) -> ray.ObjectRef:
    ref = ray.put(extract_tensors(model))

    return ref


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def to_tensor(pic) -> torch.Tensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    This function does not support torchscript.

    See :class:`~torchvision.transforms.ToTensor` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """

    default_float_dtype = torch.get_default_dtype()

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=default_float_dtype).div(255)
        else:
            return img
    # handle PIL Image
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))

    if pic.mode == "1":
        img = 255 * img
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.to(dtype=default_float_dtype).div(255)
    else:
        return img
