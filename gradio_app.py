import argparse
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn


class Generator(nn.Module):
    """Generator architecture compatible with EXP4/EXP5 notebooks."""

    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_generator(checkpoint_path: str, device: torch.device) -> tuple[Generator, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both full checkpoints and direct state_dict files.
    if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
        latent_dim = int(checkpoint.get("latent_dim", 100))
        state_dict = checkpoint["generator_state_dict"]
    elif isinstance(checkpoint, dict):
        latent_dim = 100
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    generator = Generator(latent_dim=latent_dim).to(device)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator, latent_dim


def make_image_grid(images: torch.Tensor, nrows: int = 4, ncols: int = 4) -> Image.Image:
    fig, axes = plt.subplots(nrows, ncols, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        if i < images.shape[0]:
            ax.imshow(images[i].squeeze(), cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    fig.canvas.draw()
    array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    array = array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image = Image.fromarray(array[:, :, :3])
    plt.close(fig)
    return image


def build_app(default_checkpoint: str) -> gr.Blocks:
    device = get_device()

    with gr.Blocks(title="Fashion-MNIST GAN Generator") as app:
        gr.Markdown(
            """
            # Fashion-MNIST Generator UI (Gradio)
            Generate synthetic Fashion-MNIST samples from a trained GAN/WGAN generator checkpoint.
            """
        )

        with gr.Row():
            checkpoint_input = gr.Textbox(
                label="Checkpoint path",
                value=default_checkpoint,
                placeholder="Path to .pth file",
            )
            seed_input = gr.Number(label="Seed", value=111, precision=0)

        with gr.Row():
            samples_input = gr.Slider(label="Number of samples", minimum=1, maximum=64, value=16, step=1)
            noise_scale_input = gr.Slider(label="Noise scale", minimum=0.2, maximum=2.0, value=1.0, step=0.1)

        generate_btn = gr.Button("Generate Samples", variant="primary")
        output_image = gr.Image(label="Generated Grid", type="pil")
        status_text = gr.Textbox(label="Status", interactive=False)

        def generate_samples(checkpoint_path: str, seed: float, num_samples: int, noise_scale: float):
            checkpoint = Path(checkpoint_path)
            if not checkpoint.exists():
                return None, f"Checkpoint not found: {checkpoint_path}"

            try:
                generator, latent_dim = load_generator(str(checkpoint), device)
            except Exception as exc:  # noqa: BLE001
                return None, f"Failed to load model: {exc}"

            seed_value = int(seed)
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)

            with torch.no_grad():
                z = torch.randn(int(num_samples), latent_dim, device=device) * float(noise_scale)
                fake_images = generator(z).view(int(num_samples), 1, 28, 28)

            # Convert output from [-1, 1] to [0, 1] for display.
            fake_images = (fake_images * 0.5 + 0.5).clamp(0, 1).cpu()

            side = int(np.ceil(np.sqrt(int(num_samples))))
            grid_image = make_image_grid(fake_images, nrows=side, ncols=side)
            return grid_image, f"Generated {int(num_samples)} samples on {device.type.upper()}"

        generate_btn.click(
            fn=generate_samples,
            inputs=[checkpoint_input, seed_input, samples_input, noise_scale_input],
            outputs=[output_image, status_text],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio frontend for Fashion-MNIST generator")
    parser.add_argument(
        "--checkpoint",
        default="fashion_mnist_gan_checkpoint_v1.pth",
        help="Path to generator checkpoint (.pth)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for Gradio app")
    parser.add_argument("--port", default=7860, type=int, help="Port for Gradio app")
    args = parser.parse_args()

    app = build_app(args.checkpoint)
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
