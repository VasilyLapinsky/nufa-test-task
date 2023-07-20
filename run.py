import torch
import click

from diffusers import KandinskyV22PriorPipeline, KandinskyV22Img2ImgPipeline
from diffusers.utils import load_image


@click.command()
@click.option('--content_image_path', type=str, required=True)
@click.option('--style_image_path', type=str, required=True)
@click.option('--result_image_path', type=str, default="result.jpg")
def main(content_image_path, style_image_path, result_image_path):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prior = KandinskyV22PriorPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-2-prior',
        torch_dtype=torch.float16
    ).to(device)

    decoder = KandinskyV22Img2ImgPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-2-decoder',
        torch_dtype=torch.float16
    ).to(device)

    content_image = load_image(content_image_path)
    style_image = load_image(style_image_path)

    images_texts = ["4k photo, high resolution, high detail, fine detail", content_image, style_image]
    weights = [0.1, 0.6, 0.3]
    out = prior.interpolate(
        images_texts,
        weights,
        num_inference_steps=25,
        guidance_scale=4.
    )

    result_image = decoder(
        image=content_image,
        image_embeds=out.image_embeds,
        negative_image_embeds=out.negative_image_embeds,
        height=512,
        width=512,
        num_inference_steps=100,
        guidance_scale=10,
        strength=.5
    ).images[0]

    result_image.save(result_image_path)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
