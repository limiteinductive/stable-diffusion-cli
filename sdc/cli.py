import typer
from rich.prompt import Prompt
from sdc.pipeline import run_stable_diffusion
from pathlib import Path



def main():
    prompt = Prompt.ask(":art: Enter your prompt: ", default="a digital painting of a beautiful cat with white fur artstationhd high quality 3d octane render deviantart")
    typer.echo(f"You want to generate a: {prompt}")
    batch_size = Prompt.ask(":sushi: How many images: ", default=1)

    output = run_stable_diffusion(prompt=prompt, batch_size=batch_size, device="cuda")
    

    path = Path("./.outputs")
    path.mkdir(exist_ok=True)
    for i, image in enumerate(output["sample"]):
        image.save(Path("./outputs/") / f"{i}.png")


def stable-diffusion():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)