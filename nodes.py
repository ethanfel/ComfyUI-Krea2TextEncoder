"""Text Encode (Krea2) — vision-aware conditioning for the Krea2 / K2 DiT.

Krea2 conditions on a 12-layer Qwen3-VL-4B tap (see ``comfy/text_encoders/krea2.py``).
Because that text encoder is a vision-language model, a reference image can be fed through
its *vision* path so the conditioning becomes visually informed by the image — without any
VAE / reference-latent. The Krea2 DiT (``comfy/ldm/krea2/model.py``) is pure text-to-image:
its sequence is ``[text_tokens, noisy_image_patches]`` with no slot for a reference latent,
so a VAE input would be a no-op here and is deliberately omitted.

Each reference image has an optional companion mask. When a mask is connected the image is
cropped to the mask's bounding box before the vision encoder, so the VLM only "sees" the
masked region. (This is reference-image masking; it is not inpainting — Krea2 has no
inpaint/concat pathway to regenerate a masked output region.)

This node differs from ``TextEncodeQwenImageEdit`` in two ways:
  * it forces the Krea2 *descriptor* conditioning template even when images are attached
    (the core Qwen-Edit node falls back to Qwen3-VL's plain image template), and
  * it has no VAE input, and it accepts an unbounded, auto-growing set of image+mask slots.
"""

import math
import re

import torch

import comfy.utils

# Keep in sync with the model's own template; fall back to a literal copy on non-Krea2 builds.
try:
    from comfy.text_encoders.krea2 import KREA2_TEMPLATE
except Exception:  # pragma: no cover - portability shim
    KREA2_TEMPLATE = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
        "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )


class TextEncodeKrea2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                # image1/mask1 are the seed pair; the web extension grows image2/mask2, ... on connect.
                "image1": ("IMAGE",),
                "mask1": ("MASK",),
                "vision_megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Maximum size (in megapixels) for each reference before the Qwen3-VL "
                               "vision encoder. References larger than this are downscaled; smaller "
                               "ones (e.g. a tight mask crop) are kept at native size, never upscaled.",
                }),
                "mask_padding": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.02,
                    "tooltip": "Context kept around the mask before cropping, as a fraction of the "
                               "image size added on EACH side. 0 = tight crop to the mask; 0.1 = ~10% "
                               "margin of surroundings. Only applies when a mask is connected.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "model/conditioning/krea2"
    DESCRIPTION = ("Krea2 (K2) text conditioning with optional vision prompting. Reference images are "
                   "fed through the Qwen3-VL vision path; an optional per-image mask crops the image to "
                   "the masked region. No VAE is used (Krea2 has no reference-latent pathway).")

    @staticmethod
    def _collect_indexed(kwargs, prefix):
        pattern = re.compile(r"^{}(\d+)$".format(prefix))
        out = {}
        for key, value in kwargs.items():
            match = pattern.match(key)
            if match is not None and value is not None:
                out[int(match.group(1))] = value
        return out

    @staticmethod
    def _crop_to_mask(image, mask, padding=0.0):
        """Crop image (B,H,W,C) to the mask bounding box, expanded by `padding` (a
        fraction of the image size) on each side. No-op if mask empty/None."""
        if mask is None:
            return image

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 4:  # (B,1,H,W) or similar -> (B,H,W)
            mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])

        h, w = image.shape[1], image.shape[2]
        if mask.shape[-2:] != (h, w):
            resized = comfy.utils.common_upscale(mask.unsqueeze(1), w, h, "bilinear", "disabled")
            mask = resized[:, 0]

        presence = (mask > 0.5).any(dim=0)  # collapse batch -> (H,W)
        if not bool(presence.any()):
            return image  # nothing selected: keep the whole image

        rows = torch.where(torch.any(presence, dim=1))[0]
        cols = torch.where(torch.any(presence, dim=0))[0]
        y0, y1 = int(rows[0]), int(rows[-1])
        x0, x1 = int(cols[0]), int(cols[-1])

        if padding > 0.0:  # grow the box outward for surrounding context, clamped to the image
            pad_x = round(padding * w)
            pad_y = round(padding * h)
            x0 = max(0, x0 - pad_x)
            x1 = min(w - 1, x1 + pad_x)
            y0 = max(0, y0 - pad_y)
            y1 = min(h - 1, y1 + pad_y)

        return image[:, y0:y1 + 1, x0:x1 + 1, :]

    def encode(self, clip, prompt, vision_megapixels=1.0, mask_padding=0.0, **kwargs):
        images = self._collect_indexed(kwargs, "image")
        masks = self._collect_indexed(kwargs, "mask")
        ordered = sorted(images.keys())

        images_vl = []
        image_prompt = ""
        total = int(vision_megapixels * 1024 * 1024)

        for slot, n in enumerate(ordered):
            image = self._crop_to_mask(images[n], masks.get(n), padding=mask_padding)

            samples = image.movedim(-1, 1)
            # vision_megapixels is an upper CAP, not a fixed target: only downscale
            # oversized references, never upscale. Otherwise a small mask crop gets
            # magnified to fill the VLM frame and the subject reads as huge/zoomed.
            scale_by = min(1.0, math.sqrt(total / (samples.shape[3] * samples.shape[2])))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)

            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            images_vl.append(s.movedim(1, -1)[:, :, :, :3])

            if len(ordered) > 1:
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(slot + 1)
            else:
                image_prompt += "<|vision_start|><|image_pad|><|vision_end|>"

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=KREA2_TEMPLATE)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return (conditioning,)


NODE_CLASS_MAPPINGS = {
    "TextEncodeKrea2": TextEncodeKrea2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeKrea2": "Text Encode (Krea2)",
}
