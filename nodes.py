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

# The user-facing system_prompt field holds just the system *message text*; the node wraps it in
# the chat-template scaffolding. Pull the default (Krea2's trained descriptor) out of the template
# so it stays in sync with whatever comfy ships.
_sys = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", KREA2_TEMPLATE, re.S)
KREA2_SYSTEM_DEFAULT = _sys.group(1) if _sys else (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)

# Instruct/edit-style framing (à la TextEncodeQwenImageEditPlus): paste this into system_prompt to
# make the VLM fuse the user's text WITH the reference image instead of just describing it.
# Out-of-distribution for Krea2's trained descriptor — experimental.
KREA2_INSTRUCT_SYSTEM = (
    "Describe the key features of the reference image (color, shape, size, texture, objects, "
    "background), then explain how the user's instruction should combine with or alter it, and "
    "generate a new image meeting the instruction while staying consistent with the reference "
    "where appropriate:"
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
                # system_prompt sits just above the image slots.
                "system_prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional system-instruction input. Wire a text node to override how the "
                               "VLM frames the reference + your prompt; leave unconnected to use Krea2's "
                               "trained descriptor (in-distribution). Use an instruct/edit-style "
                               "instruction (see README) to fuse the prompt with the image. The node "
                               "adds the chat-template scaffolding; provide just the instruction text.",
                }),
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
                "vision_position": (["before prompt", "after prompt"], {
                    "default": "before prompt",
                    "tooltip": "Where the image (vision) tokens sit in the user turn relative to your "
                               "text. 'before prompt' = image then text (default); 'after prompt' = text "
                               "then image. No effect without an image. Experimental.",
                }),
                "print_prompt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print the full assembled prompt sent to the Qwen3-VL encoder (system "
                               "instruction + vision placeholders + your text) to the ComfyUI console.",
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

    @classmethod
    def _prepare_vision(cls, kwargs, vision_megapixels, mask_padding):
        """Crop+resize each connected reference and build the vision-token string.
        Shared by the encoder and the VLM preview so both feed the model identically."""
        images = cls._collect_indexed(kwargs, "image")
        masks = cls._collect_indexed(kwargs, "mask")
        ordered = sorted(images.keys())

        images_vl = []
        image_prompt = ""
        total = int(vision_megapixels * 1024 * 1024)

        for slot, n in enumerate(ordered):
            image = cls._crop_to_mask(images[n], masks.get(n), padding=mask_padding)
            samples = image.movedim(-1, 1)
            # vision_megapixels is an upper CAP, not a fixed target: only downscale oversized
            # references, never upscale (a small mask crop would otherwise be magnified).
            scale_by = min(1.0, math.sqrt(total / (samples.shape[3] * samples.shape[2])))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)
            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            images_vl.append(s.movedim(1, -1)[:, :, :, :3])
            if len(ordered) > 1:
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(slot + 1)
            else:
                image_prompt += "<|vision_start|><|image_pad|><|vision_end|>"
        return images_vl, image_prompt

    @staticmethod
    def _build_text(system_prompt, prompt, image_prompt, vision_position):
        """Assemble the user text (with vision tokens) and the chat template."""
        system = system_prompt.strip() or KREA2_SYSTEM_DEFAULT
        template = ("<|im_start|>system\n" + system + "<|im_end|>\n"
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
        text = (prompt + image_prompt) if vision_position == "after prompt" else (image_prompt + prompt)
        return text, template

    @staticmethod
    def _fp8_hint(exc, images_vl):
        """Map the cryptic FP8 vision crash to an actionable error; else None.

        ComfyUI's Qwen3-VL vision tower (qwen35.py fast_pos_embed_interpolate) adds the pos-embed
        weights without casting, so an FP8-loaded text encoder dies on the image path."""
        if images_vl and isinstance(exc, NotImplementedError) and "Float8" in str(exc):
            return RuntimeError(
                "Krea2: the Qwen3-VL text encoder is loaded in FP8, which ComfyUI's vision tower "
                "cannot run on the image path ('add_stub not implemented for Float8_e4m3fn'). Load "
                "a bf16/fp16 Qwen3-VL-4B text encoder (e.g. a qwen3vl_4b *bf16* file) via CLIPLoader "
                "type 'krea2' when using image references. The FP8 encoder works only text-only."
            )
        return None

    @staticmethod
    def _quiet_token_progress():
        """Silence ComfyUI's per-token tqdm bar during generation (it prints one console line
        per token on consoles that don't honor carriage returns). The UI progress ring, driven
        by a separate comfy.utils.ProgressBar, still updates. Returns a restore() callable."""
        try:
            import comfy.text_encoders.llama as _llama
            orig = _llama.tqdm
            _llama.tqdm = lambda it=None, *a, **k: (it if it is not None else orig(*a, **k))
            return lambda: setattr(_llama, "tqdm", orig)
        except Exception:
            return lambda: None

    def encode(self, clip, prompt, vision_megapixels=1.0, mask_padding=0.0,
               system_prompt=KREA2_SYSTEM_DEFAULT, vision_position="before prompt",
               print_prompt=False, **kwargs):
        images_vl, image_prompt = self._prepare_vision(kwargs, vision_megapixels, mask_padding)
        text, template = self._build_text(system_prompt, prompt, image_prompt, vision_position)

        if print_prompt:
            print("\n========== Text Encode (Krea2) -> Qwen3-VL prompt ==========")
            print(template.replace("{}", text, 1))  # literal replace: brace-safe
            print("---- references: {} ----".format(len(images_vl)))
            print("===========================================================\n")

        tokens = clip.tokenize(text, images=images_vl, llama_template=template)
        try:
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        except NotImplementedError as exc:
            hint = self._fp8_hint(exc, images_vl)
            if hint is not None:
                raise hint from exc
            raise
        return (conditioning,)


class Krea2SystemPrompt:
    """Generic text node preloaded with the instruct/edit-style system prompt. Wire its
    output into TextEncodeKrea2's `system_prompt` input to make the prompt fuse with the
    reference image (experimental / out-of-distribution). Edit the text freely."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, "default": KREA2_INSTRUCT_SYSTEM,
                    "tooltip": "System instruction for Krea2's VLM. Defaults to an instruct/edit-style "
                               "framing that fuses your prompt with the reference image. Edit as needed; "
                               "paste the plain descriptor to fall back to default behavior.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "run"
    CATEGORY = "model/conditioning/krea2"
    DESCRIPTION = ("Text node preloaded with an instruct-style system prompt for Text Encode (Krea2). "
                   "Wire its output into the encoder's system_prompt input.")

    def run(self, text):
        return (text,)


class Krea2VLMPreview:
    """Runs Krea2's Qwen3-VL encoder GENERATIVELY on the same image+prompt the encoder feeds,
    returning the model's text output — a proxy for what the VLM 'sees'. Needs the encoder's
    lm_head in the weights; use a bf16/fp16 encoder for image inputs (FP8 vision is unsupported)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Same as the encoder's system_prompt input; unconnected = Krea2 descriptor.",
                }),
                "image1": ("IMAGE",),
                "mask1": ("MASK",),
                "vision_megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "mask_padding": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.02}),
                "vision_position": (["before prompt", "after prompt"], {"default": "before prompt"}),
                "max_length": ("INT", {"default": 256, "min": 1, "max": 8192,
                                       "tooltip": "Max number of tokens to generate."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01,
                                          "tooltip": "0 = greedy/deterministic."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("vlm_text",)
    FUNCTION = "preview"
    CATEGORY = "model/conditioning/krea2"
    DESCRIPTION = ("Generate text from Krea2's Qwen3-VL encoder on the same image+prompt the encoder "
                   "sends, to inspect how the VLM reads your reference. Needs lm_head weights; use a "
                   "bf16 encoder for image inputs.")

    def preview(self, clip, prompt, vision_megapixels=1.0, mask_padding=0.0,
                system_prompt=KREA2_SYSTEM_DEFAULT, vision_position="before prompt",
                max_length=256, temperature=0.7, seed=0, **kwargs):
        images_vl, image_prompt = TextEncodeKrea2._prepare_vision(kwargs, vision_megapixels, mask_padding)
        text, template = TextEncodeKrea2._build_text(system_prompt, prompt, image_prompt, vision_position)
        tokens = clip.tokenize(text, images=images_vl, llama_template=template)
        restore = TextEncodeKrea2._quiet_token_progress()
        try:
            ids = clip.generate(tokens, do_sample=(temperature > 0.0), max_length=max_length,
                                temperature=max(temperature, 0.01), seed=seed)
        except NotImplementedError as exc:
            hint = TextEncodeKrea2._fp8_hint(exc, images_vl)
            if hint is not None:
                raise hint from exc
            raise
        finally:
            restore()
        return (clip.decode(ids),)


NODE_CLASS_MAPPINGS = {
    "TextEncodeKrea2": TextEncodeKrea2,
    "Krea2SystemPrompt": Krea2SystemPrompt,
    "Krea2VLMPreview": Krea2VLMPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeKrea2": "Text Encode (Krea2)",
    "Krea2SystemPrompt": "Krea2 System Prompt",
    "Krea2VLMPreview": "Krea2 VLM Preview",
}
