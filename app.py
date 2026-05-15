import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional
import torch
import torchvision
from torchvision.io import read_image
import torchvision.transforms.v2.functional as tvf
from safetensors.torch import load_file
import gradio as gr
from scorer.modeling_vtoniqa import VTONScorer
from scorer.configuration_vtoniqa import VTONScorerConfig
from dataset import pad_to_square_tensor, resize_to_match


@dataclass
class AppConfig:
    scorer_model_root: str = "ckpt"
    asset_dir: str = "assets"
    img_size: int = 256
    share: bool = False
    server_name: Optional[str] = None
    server_port: Optional[int] = None


def parse_config() -> AppConfig:
    parser = argparse.ArgumentParser(description="VTON-IQA Demo")

    parser.add_argument(
        "--scorer_model_root",
        type=str,
        default=AppConfig.scorer_model_root,
        help="Path to the pretrained VTON-IQA scorer model",
    )
    parser.add_argument(
        "--asset_dir",
        type=str,
        default=AppConfig.asset_dir,
        help="Path to example asset directory",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=AppConfig.img_size,
        help="Image size for example thumbnails",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio share link",
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default=None,
        help="Server name for Gradio, e.g. 0.0.0.0",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=None,
        help="Server port for Gradio",
    )

    args = parser.parse_args()

    return AppConfig(
        scorer_model_root=args.scorer_model_root,
        asset_dir=args.asset_dir,
        img_size=args.img_size,
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


def collect_examples(asset_dir: str):
    examples = []

    if not os.path.isdir(asset_dir):
        return examples

    for example_id in sorted(os.listdir(asset_dir)):
        example_dir = os.path.join(asset_dir, example_id)

        if not os.path.isdir(example_dir):
            continue

        person_path = None
        garment_path = None

        for fname in os.listdir(example_dir):
            lower = fname.lower()

            if lower.endswith(("_0.jpg", "_0.png")):
                person_path = os.path.join(example_dir, fname)

            elif lower.endswith(("_1.jpg", "_1.png")):
                garment_path = os.path.join(example_dir, fname)

        vton_dir = os.path.join(example_dir, "vton")

        if person_path is None:
            continue
        if garment_path is None:
            continue
        if not os.path.isdir(vton_dir):
            continue

        vton_paths = []
        for fname in sorted(os.listdir(vton_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                vton_paths.append(os.path.join(vton_dir, fname))

        if len(vton_paths) == 0:
            continue

        examples.append(
            {
                "id": example_id,
                "person": person_path,
                "garment": garment_path,
                "vtons": vton_paths,
            }
        )

    return examples


class VTONScorerOnce:
    def __init__(self, scorer_model_root: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config_path = os.path.join(scorer_model_root, "config.json")
        weight_path = os.path.join(scorer_model_root, "model.safetensors")

        with open(config_path, "r") as f:
            config = VTONScorerConfig(**json.load(f))

        self.model = VTONScorer(config)
        self.model.load_state_dict(load_file(weight_path))
        self.model.to(self.device)
        self.model.eval()

    def __call__(
        self,
        person_path: str,
        garment_path: str,
        vton_path: str,
    ) -> float:
        person = read_image(
            person_path,
            mode=torchvision.io.ImageReadMode.RGB,
        )
        garment = read_image(
            garment_path,
            mode=torchvision.io.ImageReadMode.RGB,
        )
        vton = read_image(
            vton_path,
            mode=torchvision.io.ImageReadMode.RGB,
        )

        vton = tvf.resize(
            vton,
            size=person.shape[1:],
            interpolation=tvf.InterpolationMode.BICUBIC,
        )

        person = pad_to_square_tensor(person)
        garment = pad_to_square_tensor(garment)
        vton = pad_to_square_tensor(vton)

        vton = resize_to_match(vton, person, mode="bilinear")

        person = tvf.resize(
            person,
            size=(512, 512),
            interpolation=tvf.InterpolationMode.BICUBIC,
        )
        garment = tvf.resize(
            garment,
            size=(512, 512),
            interpolation=tvf.InterpolationMode.BICUBIC,
        )
        vton = tvf.resize(
            vton,
            size=(512, 512),
            interpolation=tvf.InterpolationMode.BICUBIC,
        )

        person = person.to(self.device)
        garment = garment.to(self.device)
        vton = vton.to(self.device)

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        ):
            scores = self.model(person, garment, vton)

        return scores.item()


def build_demo(config: AppConfig) -> gr.Blocks:
    scorer = VTONScorerOnce(scorer_model_root=config.scorer_model_root)
    example_data = collect_examples(config.asset_dir)

    def predict_score(
        person_path: Optional[str],
        garment_path: Optional[str],
        vton_path: Optional[str],
    ) -> str:
        if person_path is None or garment_path is None or vton_path is None:
            return "Please upload all three images."

        try:
            score = scorer(
                person_path=person_path,
                garment_path=garment_path,
                vton_path=vton_path,
            )
            return f"{score:.3f}"

        except Exception as e:
            return f"Error: {e}"

    def clear_all():
        return None, None, None, ""

    def load_example_tryon(
        person_path: str,
        garment_path: str,
        tryon_path: str,
    ):
        return person_path, garment_path, tryon_path

    with gr.Blocks(title="VTON-IQA Demo") as demo:
        gr.Markdown("# VTON-IQA Demo")
        gr.Markdown(
            "Input person image, garment image, and virtual try-on image "
            "to calculate the quality score"
        )

        with gr.Row():
            person_input = gr.Image(
                type="filepath",
                label="Person",
                sources=["upload"],
                height=512,
            )
            cloth_input = gr.Image(
                type="filepath",
                label="Garment",
                sources=["upload"],
                height=512,
            )
            tryon_input = gr.Image(
                type="filepath",
                label="Virtual Try-On",
                sources=["upload"],
                height=512,
            )

        with gr.Row():
            run_button = gr.Button(
                "Calculate Quality Score",
                variant="primary",
            )
            clear_button = gr.Button("Clear")

        score_output = gr.Textbox(
            label="Quality Score",
            interactive=False,
        )

        gr.Markdown("## Examples")

        if len(example_data) == 0:
            gr.Markdown(
                "No example images found. "
                f"Please check the structure of `{config.asset_dir}/{{id}}/...`."
            )

        for ex in example_data:
            gr.Markdown("---")

            with gr.Row():
                gr.Image(
                    value=ex["person"],
                    label="Person",
                    interactive=False,
                    height=config.img_size,
                    type="filepath",
                )

                gr.Image(
                    value=ex["garment"],
                    label="Garment",
                    interactive=False,
                    height=config.img_size,
                    type="filepath",
                )

                for i, vton_path in enumerate(ex["vtons"]):
                    with gr.Column():
                        gr.Image(
                            value=vton_path,
                            label=f"Ex. {i + 1}",
                            interactive=False,
                            height=config.img_size,
                            type="filepath",
                        )

                        use_button = gr.Button(
                            f"Ex. {i + 1}",
                            variant="secondary",
                        )

                        use_button.click(
                            fn=load_example_tryon,
                            inputs=[
                                gr.State(ex["person"]),
                                gr.State(ex["garment"]),
                                gr.State(vton_path),
                            ],
                            outputs=[
                                person_input,
                                cloth_input,
                                tryon_input,
                            ],
                        )

        run_button.click(
            fn=predict_score,
            inputs=[
                person_input,
                cloth_input,
                tryon_input,
            ],
            outputs=score_output,
        )

        clear_button.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                person_input,
                cloth_input,
                tryon_input,
                score_output,
            ],
        )

    return demo


def main() -> None:
    config = parse_config()
    demo = build_demo(config)

    launch_kwargs = {
        "share": config.share,
    }

    if config.server_name is not None:
        launch_kwargs["server_name"] = config.server_name

    if config.server_port is not None:
        launch_kwargs["server_port"] = config.server_port

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
