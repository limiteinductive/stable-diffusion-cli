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
