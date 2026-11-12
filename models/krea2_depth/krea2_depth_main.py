# Krea 2 Depth ControlNet - WanGP model plugin
#
# How the Patil/Krea-2-depth-controlnet Control LoRA works (reference: github.com/Tanmaypatil123/Krea-2-controlnet):
#   * the depth map (Depth Anything V2 Large, inverse depth, near = white) is encoded with the Krea 2 / Qwen-Image VAE;
#   * at every step its patchified latent is concatenated channel-wise to the noisy latent (64 -> 128 features per token);
#   * the DiT input projection `first` is replaced by a fully trained Linear(128 -> 6144) stored in the LoRA file;
#   * a rank-64 LoRA on attn.{wq,wk,wv,wo,gate} and mlp.{gate,up,down} of the 28 blocks steers generation. The `--lora-scale`
#     of the reference only scales these block LoRAs: here it is the "Depth Control Weight" slider.
#
# Implementation: Linear([x ; c]) = W_img x + b + W_ctrl c. The trained image half (+ bias) replaces `first` (so it stays a
# 64-feature Linear that other Krea 2 LoRAs can still target) and the control half lives in `depth_control_in`, added to the output
# of `first` by a forward hook. The Krea 2 pipeline itself is left untouched.

import os
import types

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open

from shared.utils import files_locator as fl
from shared.utils.phase_progress import generation_progress

from models.krea2.krea2_main import model_factory as _krea2_model_factory

from .krea2_depth_handler import DEPTH_LORA_URL


_FIRST_WEIGHT_KEYS = ("first.weight", "diffusion_model.first.weight", "model.diffusion_model.first.weight", "transformer.first.weight")


# ------------------------------------------------------------------------------------------------ LoRA file

def _lora_dir_for(model_type):
    try:
        from wgp import get_lora_dir  # wgp registers itself in sys.modules: this does not re-run the app

        return get_lora_dir(model_type)
    except Exception as error:  # running outside WanGP, or LoRA folder unavailable
        print(f"[Krea 2 Depth] Krea 2 LoRA folder unavailable ({error}); falling back to the models folder.")
        return None


def resolve_depth_lora_path(model_type, model_def, download=True):
    """Absolute path of the depth Control LoRA: the Krea 2 LoRA folder first, then the models (ckpts) folders. Downloaded when
    missing (normally WanGP already did it through `preload_URLs` before loading the model)."""
    url = (model_def or {}).get("krea2_depth_lora_URL", DEPTH_LORA_URL)
    file_name = os.path.basename(url)
    lora_dir = _lora_dir_for(model_type)
    candidates = []
    if lora_dir is not None:
        candidates.append(os.path.join(lora_dir, file_name))
    if os.path.isabs(url) or not url.startswith("http"):
        candidates.insert(0, url)
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    located = fl.locate_file(file_name, error_if_none=False)
    if located is not None:
        return os.path.abspath(located)
    if not download or not url.startswith("http"):
        raise FileNotFoundError(f"Krea 2 depth Control LoRA '{file_name}' not found (looked in {candidates} and the models folders).")
    target = candidates[-1] if lora_dir is not None else fl.get_download_location(file_name)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    print(f"[Krea 2 Depth] Downloading the depth Control LoRA to '{target}'")
    from shared.utils.download import download_url

    download_url(url, target)
    return os.path.abspath(target)


def load_expanded_input_projection(lora_path):
    """(weight [features, 2 * in], bias [features]) of the trained expanded `first` projection stored in the Control LoRA."""
    with safe_open(lora_path, framework="pt", device="cpu") as reader:
        keys = set(reader.keys())
        weight_key = next((key for key in _FIRST_WEIGHT_KEYS if key in keys), None)
        if weight_key is None:
            raise ValueError(f"'{os.path.basename(lora_path)}' has no expanded 'first.weight' input projection: it is not a Krea 2 Control LoRA.")
        bias_key = weight_key[: -len("weight")] + "bias"
        weight = reader.get_tensor(weight_key)
        bias = reader.get_tensor(bias_key) if bias_key in keys else None
    return weight, bias


def convert_control_lora_state_dict(state_dict):
    """Krea 2 Control LoRA keys (`blocks.N.attn.wq.A` / `.B`, plus the `first.*` projection) to MMGP names. Other LoRAs are returned
    unchanged."""
    if not any(key.endswith((".A", ".B")) for key in state_dict):
        return state_dict
    converted = {}
    for key, value in state_dict.items():
        bare = key
        for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
            if bare.startswith(prefix):
                bare = bare[len(prefix):]
                break
        if bare.startswith("first."):
            continue  # the expanded input projection is installed once, when the model is loaded
        if bare.endswith(".A"):
            converted[bare[:-2] + ".lora_A.weight"] = value
        elif bare.endswith(".B"):
            converted[bare[:-2] + ".lora_B.weight"] = value
        else:
            converted[bare] = value
    return converted


# ------------------------------------------------------------------------------------------------ transformer surgery

def _depth_first_hook(module, inputs, output):
    owner = module._krea2_depth_owner
    control_tokens = getattr(owner, "_krea2_depth_tokens", None)
    if control_tokens is None:
        return output
    if control_tokens.shape[1] != output.shape[1]:
        raise RuntimeError(f"Krea 2 Depth: the depth map gives {control_tokens.shape[1]} tokens but the image has {output.shape[1]}.")
    control = owner.depth_control_in(control_tokens.to(device=output.device, dtype=output.dtype))
    return output + control  # broadcast over the batch


def install_depth_control(transformer, weight, bias, dtype=torch.bfloat16):
    in_features = transformer.config.channels * transformer.config.patch ** 2
    out_features = transformer.config.features
    if tuple(weight.shape) != (out_features, 2 * in_features):
        raise ValueError(f"Krea 2 Depth: expected an expanded input projection of shape {(out_features, 2 * in_features)}, got {tuple(weight.shape)}.")
    first = torch.nn.Linear(in_features, out_features, bias=True, device="cpu", dtype=dtype)
    control_in = torch.nn.Linear(in_features, out_features, bias=False, device="cpu", dtype=dtype)
    with torch.no_grad():
        first.weight.copy_(weight[:, :in_features].to(dtype))
        if bias is not None:
            first.bias.copy_(bias.to(dtype))
        else:
            first.bias.copy_(transformer.first.bias.detach().to("cpu", dtype))
        control_in.weight.copy_(weight[:, in_features:].to(dtype))
    first.requires_grad_(False)
    control_in.requires_grad_(False)
    object.__setattr__(first, "_krea2_depth_owner", transformer)  # plain attribute: not registered as a submodule
    first.register_forward_hook(_depth_first_hook)
    transformer.first = first
    transformer.depth_control_in = control_in
    transformer._krea2_depth_tokens = None

    original_preprocess = transformer.preprocess_loras

    def preprocess_loras(self, model_type, sd):
        return original_preprocess(model_type, convert_control_lora_state_dict(sd))

    transformer.preprocess_loras = types.MethodType(preprocess_loras, transformer)
    return transformer


# ------------------------------------------------------------------------------------------------ depth map

def depth_image_to_tensor(control, height, width):
    """Control Image (WanGP tensor [C, F, H, W] in [-1, 1] or uint8, or a PIL image) to a (1, 3, height, width) tensor in [-1, 1]:
    grayscale, min-max normalized (near = white), as the reference pipeline feeds Depth Anything V2 maps to the VAE."""
    if not torch.is_tensor(control):
        from shared.utils.utils import convert_image_to_tensor

        control = convert_image_to_tensor(control.convert("RGB"))
    if control.dim() == 4:  # [C, F, H, W]
        control = control[:, 0]
    if control.dtype == torch.uint8:
        control = control.float().div_(255.0)
    else:
        control = control.float().add(1.0).div_(2.0)
    if control.shape[0] == 1:
        control = control.expand(3, -1, -1)
    control = control[:3].clamp_(0, 1)
    gray = (control[0] * 0.299 + control[1] * 0.587 + control[2] * 0.114)[None, None]
    if gray.shape[-2:] != (height, width):
        gray = F.interpolate(gray, size=(height, width), mode="bilinear", align_corners=False)
    gray = (gray - gray.amin()) / (gray.amax() - gray.amin()).clamp_min(1e-6)
    return gray.mul(2.0).sub(1.0).expand(-1, 3, -1, -1).contiguous()


# ------------------------------------------------------------------------------------------------ model

class depth_model_factory(_krea2_model_factory):
    def __init__(self, *args, model_type=None, model_def=None, depth_model_type=None, **kwargs):
        super().__init__(*args, model_type=model_type, model_def=model_def, **kwargs)
        self.depth_model_type = depth_model_type
        self.depth_lora_path = resolve_depth_lora_path(model_type, model_def)
        weight, bias = load_expanded_input_projection(self.depth_lora_path)
        install_depth_control(self.transformer, weight, bias, dtype=self.pipeline.dtype)
        print(f"[Krea 2 Depth] Depth Control LoRA ready: {self.depth_lora_path}")

    def get_loras_transformer(self, get_model_recursive_prop, control_net_weight=None, **kwargs):
        """The block LoRA of the Control LoRA, applied at every generation with the Depth Control Weight as multiplier."""
        try:
            weight = 1.0 if control_net_weight is None else float(control_net_weight)
        except (TypeError, ValueError):
            weight = 1.0
        if weight == 0 or self.depth_lora_path is None:
            return [], []
        return [self.depth_lora_path], [weight]

    @torch.inference_mode()
    def _encode_depth_tokens(self, control, height, width):
        pipeline = self.pipeline
        device = pipeline.runtime_device
        depth = depth_image_to_tensor(control, height, width)
        tensor = depth.unsqueeze(2).to(device=device, dtype=self.vae.dtype)  # (1, 3, 1, H, W)
        latents = self.vae.encode(tensor).latent_dist.mode()
        channels = pipeline.channels
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device).view(1, channels, 1, 1, 1).to(latents.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std, device=latents.device).view(1, channels, 1, 1, 1).to(latents.dtype)
        latents = ((latents - latents_mean) / latents_std)[:, :, 0]
        patch = self.transformer.config.patch
        return rearrange(latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch).to(device=device, dtype=pipeline.dtype)

    @generation_progress
    def generate(self, input_frames=None, input_masks=None, width=1024, height=1024, callback=None, set_progress_status=None, video_prompt_type="", VAE_tile_size=None, **kwargs):
        if input_frames is None:
            raise Exception("Krea 2 Depth ControlNet needs a Control Image (a photo / render, or a depth map).")
        width, height = int(width), int(height)
        if VAE_tile_size is not None and hasattr(self.vae, "use_tiling"):
            tiling = VAE_tile_size > 0 if isinstance(VAE_tile_size, int) else bool(VAE_tile_size[0])
            tile_size = max(VAE_tile_size, 0) if isinstance(VAE_tile_size, int) else (VAE_tile_size[1] if len(VAE_tile_size) > 1 else 0)
            if tiling:
                self.vae.enable_tiling(tile_sample_min_height=tile_size or None, tile_sample_min_width=tile_size or None)
            else:
                self.vae.disable_tiling()
        if callable(set_progress_status):
            set_progress_status("Encoding Depth Map")
        self.transformer._krea2_depth_tokens = self._encode_depth_tokens(input_frames, height, width)
        try:
            # the Control Image is consumed here: Krea 2 itself runs plain text-to-image, the hook on `first` injects the depth
            return super().generate(
                input_frames=None,
                input_masks=None,
                width=width,
                height=height,
                callback=callback,
                set_progress_status=set_progress_status,
                video_prompt_type=video_prompt_type,
                VAE_tile_size=VAE_tile_size,
                **kwargs,
            )
        finally:
            self.transformer._krea2_depth_tokens = None
