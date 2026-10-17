"""LFM2-VL pictures: the checkpoint's SigLIP2 NaFlex tower and projector (transformers' own modules, bf16), and its
processor, which tiles a picture and writes the placeholder tokens. A picture's features replace its placeholders'
embeddings in the trunk; the language model's tree forward does the rest (Liquid AI's ``runner.py`` does the same)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

MAX_PIXELS = 1024 * 1024           # every picture is bounded at this many pixels first (bicubic), as runner.py does


def cap(image):
    w, h = image.size
    if w * h <= MAX_PIXELS:
        return image
    from PIL import Image

    s = math.sqrt(MAX_PIXELS / (w * h))
    return image.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.BICUBIC)


class Vision:
    def __init__(self, processor, tower, projector, image_token: int):
        self.processor, self.tower, self.projector, self.image_token = processor, tower, projector, int(image_token)
        self._markup: dict[int, str] = {}

    @classmethod
    def load(cls, model_dir: Path, device="cuda", processor: Any = None) -> Vision:
        from safetensors import safe_open
        from transformers import Lfm2VlConfig
        from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlMultiModalProjector
        from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionModel

        raw = json.loads((Path(model_dir) / "config.json").read_text())
        cfg = Lfm2VlConfig(**{k: v for k, v in raw.items() if k not in ("auto_map", "architectures")})
        tower = Siglip2VisionModel(cfg.vision_config).to(torch.bfloat16)
        projector = Lfm2VlMultiModalProjector(cfg).to(torch.bfloat16)
        towers, projections = {}, {}
        for path in sorted(Path(model_dir).glob("*.safetensors")):
            with safe_open(str(path), "pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("model.vision_tower."):
                        towers[key[len("model.vision_tower."):]] = f.get_tensor(key)
                    elif key.startswith("model.multi_modal_projector."):
                        projections[key[len("model.multi_modal_projector."):]] = f.get_tensor(key)
        if not towers:
            raise ValueError(f"no vision tower tensors in {model_dir}, which its config's vision_config needs")
        odd = next((k for k, v in (*towers.items(), *projections.items()) if v.dtype != torch.bfloat16), None)
        if odd is not None:
            raise ValueError(f"{odd} is stored as a dtype other than bf16, which the LFM2-VL tower reads")
        if not any(k in tower.state_dict() for k in towers):     # transformers releases differ in the wrapper prefix
            towers = {k.removeprefix("vision_model.") if k.startswith("vision_model.") else f"vision_model.{k}": v
                      for k, v in towers.items()}
        missing = tower.load_state_dict(towers, strict=False).missing_keys
        if missing:
            raise ValueError(f"vision tower weights missing: {missing[:4]}")
        projector.load_state_dict(projections)
        if processor is None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(str(model_dir))
        return cls(processor, tower.to(device).eval(), projector.to(device).eval(), cfg.image_token_id)

    def markup(self, n: int) -> str:
        """What the processor's chat template writes for ``n`` pictures at the head of a user turn."""

        if n not in self._markup:
            msgs = [{"role": "user", "content": [*([{"type": "image"}] * n), {"type": "text", "text": "\x00"}]}]
            text = self.processor.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
            head = "<|im_start|>user\n"
            self._markup[n] = text[text.index(head) + len(head):text.index("\x00")]
        return self._markup[n]

    @torch.inference_mode()
    def trunk(self, model, prefix: str, images: list) -> tuple[list[int], torch.Tensor]:
        """The token ids of a state's prefix with its pictures expanded (``prefix`` holds their markup), and their
        embeddings with the pictures' features in the placeholder slots."""

        inputs = self.processor(text=[prefix], images=[images], return_tensors="pt", add_special_tokens=False)
        ids = inputs["input_ids"][0]
        mask = inputs["pixel_attention_mask"]
        n = int(mask.sum(1).max())           # every picture is padded to the patch budget; the tower masks it out
        device = model.device
        out = self.tower(pixel_values=inputs["pixel_values"][:, :n].to(device, torch.bfloat16),
                         spatial_shapes=inputs["spatial_shapes"].to(device), pixel_attention_mask=mask[:, :n].to(device),
                         return_dict=True).last_hidden_state
        features = []
        for i, length in enumerate(mask.sum(1).tolist()):
            h, w = (int(v) for v in inputs["spatial_shapes"][i])
            projected = self.projector(out[i, :length].reshape(1, h, w, -1))
            features.append(projected.reshape(-1, projected.shape[-1]))
        found = torch.cat(features).to(model.embed.dtype)
        embeds = model.embed[ids.to(device)].clone()
        slots = ids.to(device) == self.image_token
        if int(slots.sum()) != found.shape[0]:
            raise ValueError(f"{int(slots.sum())} picture slots for {found.shape[0]} picture features")
        embeds[slots] = found
        return ids.tolist(), embeds
