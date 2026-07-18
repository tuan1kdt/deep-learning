"""ViVLM = SigLIP (frozen) + projector + GPT tự pretrain.

Fusion kiểu LLaVA: 49 embedding ảnh prepend vào trước embedding text; từ đó
self-attention của GPT tự attend chéo ảnh<->chữ, không cần cross-attention riêng.
"""
import torch
import torch.nn as nn


class SiglipAdapter(nn.Module):
    def __init__(self, name="google/siglip-base-patch16-224"):
        super().__init__()
        from transformers import SiglipVisionModel   # lazy import
        self.m = SiglipVisionModel.from_pretrained(name)
        self.m.requires_grad_(False)
        self.m.eval()

    @torch.no_grad()
    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).last_hidden_state


class ViVLM(nn.Module):
    def __init__(self, gpt, vision_encoder, projector):
        super().__init__()
        self.gpt = gpt
        self.vision_encoder = vision_encoder
        self.projector = projector

    def _fuse(self, pixel_values, input_ids):
        img = self.projector(self.vision_encoder(pixel_values))   # (B,49,d)
        tok = self.gpt.tok_emb(input_ids)
        return torch.cat([img, tok], dim=1), img.size(1)

    def forward(self, pixel_values, input_ids, labels=None):
        x, n_img = self._fuse(pixel_values, input_ids)
        targets = None
        if labels is not None:
            pad = torch.full((labels.size(0), n_img), -100,
                             dtype=labels.dtype, device=labels.device)
            targets = torch.cat([pad, labels], dim=1)
        return self.gpt.forward_from_embeds(x, targets)

    @torch.no_grad()
    def generate(self, pixel_values, prompt_ids, max_new_tokens,
                 temperature=0.0, top_p=None, eos_id=None):
        import torch.nn.functional as F
        ids = prompt_ids
        for _ in range(max_new_tokens):
            x, _ = self._fuse(pixel_values, ids)
            logits, _ = self.gpt.forward_from_embeds(x)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_p is not None:
                    sl, si = torch.sort(logits, descending=True)
                    probs = F.softmax(sl, dim=-1)
                    mask = probs.cumsum(-1) - probs > top_p
                    sl[mask] = float("-inf")
                    logits = torch.full_like(logits, float("-inf")) \
                                  .scatter(1, si, sl)
                next_id = torch.multinomial(F.softmax(logits, -1), 1)
            ids = torch.cat([ids, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return ids[:, prompt_ids.size(1):]

    def trainable_parameters(self, phase):
        if phase == "projector":
            return list(self.projector.parameters())
        if phase == "full":
            return list(self.projector.parameters()) + list(self.gpt.parameters())
        raise ValueError(phase)
