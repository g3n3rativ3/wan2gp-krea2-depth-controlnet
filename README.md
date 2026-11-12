# Krea 2 Depth ControlNet — WanGP model plugin

Adds two depth-guided Krea 2 models to WanGP, based on the
[Patil/Krea-2-depth-controlnet](https://huggingface.co/Patil/Krea-2-depth-controlnet) Control LoRA:

| Model | Base | Steps | CFG |
|---|---|---|---|
| **Krea 2 Turbo Depth ControlNet** | Krea 2 Turbo | 8 | 0 |
| **Krea 2 RAW Depth ControlNet** | Krea 2 RAW (the LoRA was trained on it) | 28–52 | 3.5 |

Give it a photo or render plus a prompt: the new image keeps the 3D structure, perspective and composition
of the photo, while content and style follow the prompt.

## Installation

1. Copy the `wan2gp-krea2-depth-controlnet` folder into WanGP's `plugins/` folder
   (or install it from its GitHub URL in **Apps & PlugIn Manager → Install from URL**).
2. In **Apps & PlugIn Manager**, enable *Krea 2 Depth ControlNet*, then restart WanGP.
3. The two models appear in the **Krea 2** family.

Nothing to download by hand:

- the Krea 2 transformer, text encoder and VAE are the ones WanGP already uses for Krea 2 (shared files);
- `depth-control-lora.safetensors` (862 MB) is downloaded automatically the first time a depth model is loaded,
  into the **Krea 2 LoRA folder** (`loras/krea2/` by default). If that folder can't be resolved, it goes to the
  models folder (`ckpts/`). If you already have the file in either place, it is reused.
- Depth Anything V2 is WanGP's own depth preprocessor (downloaded by WanGP the first time it is used).

## Usage

1. Select **Krea 2 Turbo Depth ControlNet**.
2. **Depth Control**:
   - *Transfer Depth (computed from the Control Image)*: upload any photo/render; WanGP extracts its depth map.
   - *Control Image is already a Depth Map*: upload your own depth map (near = white, far = black).
3. Write the prompt (it may be left empty: the depth map is then the only signal).
4. **Depth Control Weight** (default 1.0) = the reference `--lora-scale`: 0.6 is loose, 1.2 is strict.
   Two presets are provided under **Presets**.

Tips: images with real perspective work best (flat 2D illustrations give almost uniform depth maps).
The LoRA was trained around 1 megapixel: 1024×1024, 1152×896, 1216×832, 1344×768.
For the closest match with training, keep **Configuration → Depth Anything** on the default `vitl` (Large) variant.

## How it works

The reference pipeline concatenates the VAE-encoded depth latent channel-wise to the noisy latent
(64 → 128 features per token) and replaces the DiT input projection `first` with a trained `Linear(128 → 6144)`,
plus a rank-64 LoRA on all 28 blocks. Because `Linear([x ; c]) = W_img·x + b + W_ctrl·c`, the plugin:

- replaces `first` with the trained image half + bias (still a 64-feature Linear, so other Krea 2 LoRAs keep working),
- adds the control half as `depth_control_in`, added to the output of `first` by a forward hook,
- loads the block LoRA through WanGP/MMGP at each generation, with the Depth Control Weight as multiplier
  (the `.A` / `.B` keys of the checkpoint are converted to MMGP names on the fly).

The built-in Krea 2 pipeline (text encoding, sampler, Turbo `mu = 1.15`, RAW CFG, NAG, VAE upsamplers…) is reused unchanged.

## Notes

- The depth LoRA lives in the Krea 2 LoRA folder, so it also shows up in the LoRA list of the Krea 2 models.
  Don't activate it manually: the depth models apply it automatically, and it is useless on the other Krea 2 models.
- Weights: Krea 2 community license. Depth Control LoRA by @Tanmaypatil79, @Shauray7, @edwixxxx.
