# Krea 2 Depth ControlNet - WanGP model plugin
# Adds depth-conditioned Krea 2 models using the Patil/Krea-2-depth-controlnet Control LoRA.

import gradio as gr
import torch

from models.krea2.krea2_handler import family_handler as _krea2_handler


TURBO_DEPTH_MODEL_TYPE = "krea2_turbo_depth"
RAW_DEPTH_MODEL_TYPE = "krea2_raw_depth"
# each depth architecture runs on top of the matching built-in Krea 2 checkpoint
UNDERLYING_MODEL_TYPES = {TURBO_DEPTH_MODEL_TYPE: "krea2_turbo", RAW_DEPTH_MODEL_TYPE: "krea2_raw"}

DEPTH_LORA_URL = "https://huggingface.co/Patil/Krea-2-depth-controlnet/resolve/main/depth-control-lora.safetensors"
_PRESET_PROFILE_DIRS = ["krea2_depth_presets", "krea2_presets"]

_INFOS = """Generate an image that keeps the **3D structure, perspective and composition** of a **Control Image** while its content and style follow the Text Prompt (`prompt`).

- **Transfer Depth**: upload any photo or render. WanGP extracts its depth map with Depth Anything V2 and the depth Control LoRA steers Krea 2 toward it.
- **Control Image is already a Depth Map**: supply your own depth map (near = white, far = black).
- **Depth Control Weight** (default 1.0) is the structure-adherence dial: below 1 gives more creative freedom (0.6 is a good loose setting), above 1 tightens the structure at some cost to quality.

Photos and renders with real perspective work best; flat 2D illustrations give nearly uniform depth maps and weak control. The LoRA was trained around 1 megapixel: 1024x1024, 1152x896, 1216x832 or 1344x768 work best. An empty prompt is allowed: the depth map is then the only signal.

Depth Control LoRA by Tanmay Patil and collaborators ([Patil/Krea-2-depth-controlnet](https://huggingface.co/Patil/Krea-2-depth-controlnet)), weights under the Krea 2 community license."""

_PROMPT_INFOS = 'Describe the finished image in natural language: subject, materials, setting, lighting and style. The layout and perspective come from the Control Image, so focus on *what* the scene should look like rather than where things are, for example: "a futuristic spaceship interior, brushed metal walls, cinematic blue lighting". Leave the prompt empty to see how much structure the depth map alone carries.'

_DEEPY_INFOS = "Depth-controlled Krea 2: `video_guide` is a Control Image whose depth/3D layout is kept while `prompt` sets content and style. `video_prompt_type` `DV` extracts depth with Depth Anything V2, `V` means the image is already a depth map (near = white). `control_net_weight` (0-2, default 1) sets structure adherence: 0.6 = loose, 1.2 = strict. Best around 1 MP."

_DEEPY_PROMPT_INFOS = "Describe the materials, subject, lighting and style of the finished scene; the composition comes from the depth map. An empty prompt is valid."


def _underlying(base_model_type):
    return UNDERLYING_MODEL_TYPES.get(base_model_type, "krea2_turbo")


class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        result = _krea2_handler.query_model_def(_underlying(base_model_type), model_def)
        # no inpainting / LanPaint: the Control Image is a depth source, not an image to edit
        for key in ("guide_custom_choices_image", "model_modes", "inpaint_video_prompt_type", "inpaint_color"):
            result.pop(key, None)
        result.update({
            "inpaint_support": False,
            "guide_preprocessing": {
                "selection": ["DV", "V"],
                "labels": {
                    "DV": "Transfer Depth (computed from the Control Image)",
                    "V": "Control Image is already a Depth Map (near = white)",
                },
                "default": "DV",
                "label": "Depth Control",
            },
            "mask_preprocessing": {"selection": [""], "visible": False},
            "control_net_weight_name": "Depth Control",
            "control_net_weight_size": 1,
            # downloaded with the model files, straight into the Krea 2 LoRA folder
            "preload_URLs": [DEPTH_LORA_URL + "|%lora_dir"],
            "krea2_depth_lora_URL": DEPTH_LORA_URL,
            "preset_profiles_dir": list(_PRESET_PROFILE_DIRS),
            "infos": _INFOS,
            "prompt_infos": _PROMPT_INFOS,
            "deepy_infos": _DEEPY_INFOS,
            "deepy_prompt_infos": _DEEPY_PROMPT_INFOS,
        })
        return result

    @staticmethod
    def query_supported_types():
        return [TURBO_DEPTH_MODEL_TYPE, RAW_DEPTH_MODEL_TYPE]

    @staticmethod
    def query_family_maps():
        compatible = [TURBO_DEPTH_MODEL_TYPE, RAW_DEPTH_MODEL_TYPE]
        return {}, {model_type: compatible for model_type in compatible}

    @staticmethod
    def query_model_family():
        return "krea2"

    @staticmethod
    def query_family_infos():
        return {"krea2": (1150, "Krea 2")}

    @staticmethod
    def get_lora_dir(base_model_type):
        return "krea2"  # same LoRA folder as the built-in Krea 2 models

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return _krea2_handler.query_model_files(computeList, _underlying(base_model_type), model_def)

    @staticmethod
    def load_model(
        model_filename,
        model_type=None,
        base_model_type=None,
        model_def=None,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        save_quantized=False,
        submodel_no_list=None,
        text_encoder_filename=None,
        VAE_upsampling=None,
        **kwargs,
    ):
        from .krea2_depth_main import depth_model_factory

        pipe_processor = depth_model_factory(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type=model_type,
            model_def=model_def,
            base_model_type=_underlying(base_model_type),
            depth_model_type=base_model_type,
            text_encoder_filename=text_encoder_filename,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            VAE_upsampling=VAE_upsampling,
            save_quantized=save_quantized,
        )
        pipe = {
            "transformer": pipe_processor.transformer,
            "text_encoder": pipe_processor.text_encoder.language_model,
            "vae": pipe_processor.vae,
        }
        return pipe_processor, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "image_mode": 1,
            "batch_size": 1,
            "video_prompt_type": "DV",
            "control_net_weight": 1.0,
            "denoising_strength": 1.0,
            "masking_strength": 1.0,
            "resolution": "1024x1024",
        })
        if base_model_type == TURBO_DEPTH_MODEL_TYPE:
            ui_defaults.update({"num_inference_steps": 8, "guidance_scale": 0})
        else:
            ui_defaults.update({"num_inference_steps": 28, "guidance_scale": 3.5})

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        ui_defaults["image_mode"] = 1
        video_prompt_type = ui_defaults.get("video_prompt_type", "") or ""
        if "V" not in video_prompt_type:
            ui_defaults["video_prompt_type"] = "DV"
        ui_defaults.setdefault("control_net_weight", 1.0)

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        # WanGP itself checks that a Control Image was supplied once "V" is selected
        video_prompt_type = inputs.get("video_prompt_type", "") or ""
        if "V" not in video_prompt_type:
            return "Krea 2 Depth ControlNet needs a Control Image: choose 'Transfer Depth' (photo / render) or 'Control Image is already a Depth Map'."
        try:
            weight = float(inputs.get("control_net_weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if weight > 1.5:
            gr.Info("Depth Control Weight above 1.5 may degrade image quality.")
        return None

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("qwen")
