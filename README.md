# ComfyUI-nunchaku-rocm

ComfyUI nodes that run [Nunchaku](https://github.com/nunchaku-ai/nunchaku) 4-bit SVDQuant models on AMD GPUs, on
[nunchaku-rocm](https://github.com/TwelveNights/nunchaku-rocm) (Triton kernels, plus native int4 kernels on RDNA3):
FLUX.1, FLUX.2 [klein], Qwen-Image, Z-Image-Turbo and SDXL.

Only the diffusion model is replaced. Text encoders, VAE, sampler and scheduler are ComfyUI's own nodes, so the usual
workflows apply with one node swapped.

| Workflow (1024², warm) | RX 7800 XT 16 GB, Windows 11 |
| --- | --- |
| FLUX.2 [klein] 4B, 4 steps | 2.9 s |
| FLUX.1-schnell, 4 steps | 5.5-5.7 s |
| Z-Image-Turbo, 8 steps | 8.4-8.5 s |
| Qwen-Image Lightning, 8 steps | 11.6 s |
| SDXL, 30 steps | 10.6-10.7 s |

Measurements, VRAM notes and accuracy checks against ComfyUI's bf16 models are in
[nunchaku-rocm's ComfyUI notes](https://github.com/TwelveNights/nunchaku-rocm/blob/main/comfyui/README.md).

## Nodes

Category *Nunchaku (ROCm)*:

- **Nunchaku DiT Loader (ROCm)** loads a Nunchaku checkpoint from `models/diffusion_models` as a `MODEL`. The model
  type is read from the checkpoint. Qwen-Image only: `cpu_offload` (`auto` / `enable` / `disable`) streams the
  transformer blocks from CPU memory, on by default below 20 GiB of VRAM; `num_blocks_on_gpu` keeps more blocks
  resident.
- **Nunchaku LoRA Loader (ROCm)** adds a LoRA to a FLUX.1 / FLUX.2 [klein] model (chain several for more).
  ComfyUI's own LoraLoader cannot reach the 4-bit layers.

## Install

1. Install [nunchaku-rocm](https://github.com/TwelveNights/nunchaku-rocm#install) into ComfyUI's Python environment,
   on top of a ROCm PyTorch. This node pack does not install it, so it never touches your PyTorch.
2. Install this node pack with ComfyUI Manager, or clone it into `ComfyUI/custom_nodes`:

   ```
   git clone https://github.com/TwelveNights/ComfyUI-nunchaku-rocm.git
   ```

3. Restart ComfyUI and open a workflow from [`example_workflows`](example_workflows).

The first image of each model compiles and autotunes kernels and can take a minute or two; later runs use the cache.

## Models

Use the **int4** checkpoints: AMD GPUs have no FP4 matrix units, so NVFP4 files run on a slower emulated path.

| Model | Checkpoint (`models/diffusion_models`) | Text encoder(s) (`models/text_encoders`) | VAE (`models/vae`) |
| --- | --- | --- | --- |
| FLUX.2 [klein] 4B | `tonera/FLUX.2-klein-4B-Nunchaku`: `svdq-int4_r32-…` | `qwen_3_4b.safetensors` (type `flux2`) | `flux2-vae.safetensors` |
| FLUX.2 [klein] 9B | `tonera/FLUX.2-klein-9B-Nunchaku`: `svdq-int4_r32-…` | `qwen_3_8b_fp8mixed.safetensors` (type `flux2`) | `flux2-vae.safetensors` |
| FLUX.1-schnell / dev | `nunchaku-ai/nunchaku-flux.1-schnell` (or `-dev`): `svdq-int4_r32-…` | `clip_l` + `t5xxl_fp16` (DualCLIPLoader, type `flux`) | `ae.safetensors` |
| Qwen-Image, Lightning | `nunchaku-ai/nunchaku-qwen-image`: `svdq-int4_r32-qwen-image…` | `qwen_2.5_vl_7b_fp8_scaled.safetensors` (type `qwen_image`) | `qwen_image_vae.safetensors` |
| Z-Image-Turbo | `nunchaku-ai/nunchaku-z-image-turbo`: `svdq-int4_r32-z-image-turbo` | `qwen_3_4b.safetensors` (type `lumina2`) | `ae.safetensors` |
| SDXL, SDXL-Turbo | `nunchaku-ai/nunchaku-sdxl`, `nunchaku-ai/nunchaku-sdxl-turbo` | `clip_l` + `clip_g` (DualCLIPLoader, type `sdxl`) | `sdxl_vae.safetensors` |

Not supported yet: ControlNet, Qwen-Image-Edit and reference latents for Qwen-Image / Z-Image, LoRA for Qwen-Image /
Z-Image / SDXL, and model-patching nodes that replace transformer blocks or attention (ignored with a warning).

## License

Apache License 2.0, like Nunchaku and nunchaku-rocm.
