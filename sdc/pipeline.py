from dataclasses import dataclass, fields
from os import strerror
from pathlib import Path
from typing import List, Tuple, TypedDict, Union

import numpy as np
import torch
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from PIL import Image
from ray import ObjectRef
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from transformers import CLIPFeatureExtractor, PreTrainedTokenizer

from sdc.utils import load_from_plasma, numpy_to_pil, performance, push_model_to_plasma, seed_everything


@dataclass
class StableDiffusionPlasma:
    vae: ObjectRef
    text_encoder: ObjectRef
    unet: ObjectRef
    safety_checker: ObjectRef
    feature_extractor: CLIPFeatureExtractor
    tokenizer: PreTrainedTokenizer


def initialize_plasma(checkpoint_path="./stable-diffusion.pt") -> StableDiffusionPlasma:
    """
    Initialize the plasma with the checkpoint.
    """
    plasma = {}
    model_dict = torch.load(checkpoint_path)

    for field, model in model_dict.items():
        plasma[field] = push_model_to_plasma(model) if isinstance(model, torch.nn.Module) else model

    return StableDiffusionPlasma(**plasma)


def process_image(image: Union[str, Image.Image]) -> torch.Tensor:
    init_image = Image.open(init_image).convert("RGB") if isinstance(init_image, str) else init_image
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


class StableDiffusionOutput(TypedDict):
    sample: List[Image.Image]
    nsfw_content_detected: List[bool]
    plasma: StableDiffusionPlasma


@performance
def run_stable_diffusion(
    prompt: str = "",
    scheduler: Union[DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler] = None,
    batch_size: int = 1,
    steps: int = 50,
    skip_steps: float = 0.0,
    width: int = 512,
    height: int = 512,
    guidance_scale: float = 7.5,
    init_image: str = None,
    seed: int = None,
    device: str = "cuda",
    plasma: StableDiffusionPlasma = None,
    checkpoint_path="./models/",
) -> StableDiffusionOutput:
    """
    Run the stable diffusion pipeline.
    """
    seed = seed_everything(seed)

    if plasma is None:
        plasma = initialize_plasma(checkpoint_path=checkpoint_path)

    if scheduler is None:
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        offset=0
        scheduler.set_timesteps(steps)
        if init_image:
            scheduler = PNDMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                skip_prk_steps=True,
            )
            offset = 1
            scheduler.set_timesteps(steps, offset=offset)

    width, height = map(lambda x: (x // 8) * 8, (width, height))

    text_input = plasma.tokenizer(
        [prompt] * batch_size,
        padding="max_length",
        max_length=plasma.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_input = plasma.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=text_input.input_ids.shape[-1],
        return_tensors="pt",
    )

    with load_from_plasma(plasma.text_encoder, device=device) as text_encoder:
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    skip_timestep = int(steps * (1 - skip_steps)) + offset
    skip_timestep = min(skip_timestep, steps)
    timesteps = scheduler.timesteps[-skip_timestep]
    timesteps = torch.tensor([timesteps] * batch_size, device=device)
    t_start = int(steps - skip_timestep)

    if init_image:
        init_image = Image.open(init_image).convert("RGB")
        init_image = init_image.resize((width, height))
        init_image = process_image(init_image)

        with load_from_plasma(plasma.vae, device=device) as vae:
            init_latents = vae.encode(init_image.to(device)).sample()
            init_latents = 0.18215 * init_latents
            init_latents = torch.cat([init_latents] * batch_size)
            noise = torch.randn(init_latents.shape, device=device)
            latents = scheduler.add_noise(init_latents, noise, timesteps)
    else:
        latents = torch.randn(
            batch_size,
            4,
            height // 8,
            width // 8,
            device=device,
        )

    if isinstance(scheduler, LMSDiscreteScheduler):
        latents = latents * scheduler.sigmas[0]

    for i, t in tqdm(enumerate(scheduler.timesteps[t_start:])):
        with load_from_plasma(plasma.unet, device=device) as unet:
            latent_model_input = torch.cat([latents] * 2)
            if isinstance(scheduler, LMSDiscreteScheduler):
                latent_model_input = latent_model_input / ((scheduler.sigmas[i] ** 2 + 1) ** 0.5)

            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if isinstance(scheduler, LMSDiscreteScheduler):
                latents = scheduler.step(noise_pred, i, latents)["prev_sample"]
            else:
                latents = scheduler.step(noise_pred, t, latents)["prev_sample"]

    with load_from_plasma(plasma.vae, device=device) as vae:
        latents = 1 / 0.18215 * latents
        images = vae.decode(latents)
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        pil_images = numpy_to_pil(images)

    with load_from_plasma(plasma.safety_checker, device=device) as safety_checker:
        safety_checker_input = plasma.feature_extractor(pil_images, return_tensors="pt").to(device)
        output_images, has_nsfw_concept = safety_checker(clip_input=safety_checker_input.pixel_values, images=images)

    return {
        "sample": pil_images,
        "nsfw_content_detected": has_nsfw_concept,
        "plasma": plasma,
    }


if __name__ == "__main__":
    output = run_stable_diffusion(
        prompt="san goku",
        init_image="outputs/5.png",
        skip_steps=0.35,
        device="cuda:2",
        checkpoint_path="/home/selas/laiogen/stable-diffusion.pt",
        batch_size=1,
    )

    for i, image in enumerate(output["sample"]):
        image.save(Path("./outputs/") / f"{i}.png")
