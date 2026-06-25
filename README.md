# ComfyUI-Krea2TexTEncoder

![Text Encode (Krea2)](assets/social.png)

**Text Encode (Krea2)** — vision-aware text conditioning for the Krea2 / K2 model
(`kreaturbo.safetensors`), whose text encoder is **Qwen3-VL-4B** (12-layer tap).

## Why this node exists

Krea2's text encoder is a vision-language model, so a reference image can be pushed
through its **vision path** to make the conditioning visually aware of that image — a
"prompt from a picture" effect. People were reaching for the core
`TextEncodeQwenImageEdit` node to do this, which has two problems for Krea2:

1. **The VAE input does nothing.** Krea2's DiT (`comfy/ldm/krea2/model.py`,
   `SingleStreamDiT`) builds its token sequence as `[text_tokens, noisy_image_patches]`
   — there is **no slot for a reference latent**, and `Krea2.extra_conds`
   (`comfy/model_base.py`) never reads `reference_latents`. So a connected VAE produces a
   `reference_latents` entry the model silently discards. (Real pixel-faithful editing
   would require training a reference-latent pathway into the DiT — it can't be done from
   a node alone.)
2. **Wrong template with images.** With an image attached, the core node falls back to
   Qwen3-VL's *plain* image template instead of the Krea2 *descriptor* template the model
   was conditioned with.

This node fixes both: it **forces the Krea2 descriptor template** even with images, and
it **omits the VAE input** entirely. It also accepts an **unbounded, auto-growing set of
image+mask pairs** (connect `image1`, a fresh `image2`/`mask2` pair appears, and so on).

## Masks

Each reference image has an optional companion mask. When connected, the image is
**cropped to the mask's bounding box** before the vision encoder, so the VLM only sees the
masked region. Use **`mask_padding`** to keep surrounding context: it grows the crop box by
that fraction of the image size on each side (`0` = tight crop, `0.1` ≈ 10% margin, high
values ≈ the whole image). The mask is used only to compute the crop — it is not itself
sent to the VLM. This is *reference-image* masking — **not inpainting**: Krea2 has no
concat/inpaint pathway to regenerate a masked region of the output. (To spatially restrict
where the *prompt* applies in the output, use ComfyUI's standard `ConditioningSetMask`
downstream of this node — that works generically at the sampler level.)

## Inputs

| Input | Type | Notes |
|-------|------|-------|
| `clip` | CLIP | Load with **CLIPLoader → type `krea2`**. |
| `prompt` | STRING | Your text prompt. |
| `image1…N` | IMAGE | Optional reference images; slots grow as you connect them. |
| `mask1…N` | MASK | Optional per-image mask; crops `imageN` to the masked bounding box. |
| `mask_padding` | FLOAT | Context kept around the mask, as a fraction of image size per side (`0` = tight, default). |
| `vision_megapixels` | FLOAT | Max size before the vision encoder; references are downscaled to this cap, never upscaled (default `1.0`). |

Output is a standard `CONDITIONING` for the Krea2 sampler. With no images connected it
works as a plain Krea2 text encoder.

## Install

Symlink (or copy) this repo into ComfyUI's `custom_nodes/`, matching the existing setup:

```bash
ln -s /media/p5/ComfyUI-Krea2TexTEncoder /media/p5/Comfyui/custom_nodes/ComfyUI-Krea2TexTEncoder
```

Then restart ComfyUI.
