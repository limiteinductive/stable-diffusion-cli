import subprocess
import typer
from rich.prompt import Prompt
from sdc.pipeline import run_stable_diffusion
from pathlib import Path


app = typer.Typer()


@app.command()
def main():
    prompt = Prompt.ask(
        ":art: Enter your prompt",
        default="A Swirling Colorful Galaxy, Intricate Detailed Color Smashing, Particles, Technologic, Ice, Broken Mirror, Fluid Acrilic, Thin Fractal Tendrils, Elegant, Ornametrical, Ruan Jia, Wlop. Scifi, Fantasy, Hyper Detailed, Octane Render, Concept Art, By Peter Mohrbacher",
    )

    batch_size = Prompt.ask(":sushi: How many images", default=1)

    checkpoint_path = Path("~/.cache/stable-diffusion/stable-diffusion.pt")
    if not checkpoint_path.exists():
        typer.echo(f"Downloading stable-diffusion checkpoint to {checkpoint_path}")
        subprocess.run(["wget", "-P", "~/.cache/stable-diffusion/", "https://storage.googleapis.com/laion_limiteinducive/stable-diffusion.pt"])

    output = run_stable_diffusion(prompt=prompt, batch_size=int(batch_size), device="cuda", checkpoint_path=checkpoint_path)

    path = Path("./.outputs")
    path.mkdir(exist_ok=True)
    for i, image in enumerate(output["sample"]):
        image.save(Path("./outputs/") / f"{i}.png")

