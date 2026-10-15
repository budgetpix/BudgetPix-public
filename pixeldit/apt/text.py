"""Caption -> Gemma embeddings.

  encoder     `Efficient-Large-Model/gemma-2-2b-it` decoder stack in bfloat16, last hidden state;
  tokenizer   the same repo, `padding_side = "right"`;
  CHI prompt  every caption is prefixed with the "complex human instruction" block, and the padded
              length is len(tokenize(chi)) + model_max_length - 2 ([bos] + [_]);
  selection   the bos token, then the last 299 positions: 300 tokens of context;
  mask        the attention mask gathered by the same indices.

The null embedding for classifier-free guidance and caption dropout: see `null_embedding`.
"""
from __future__ import annotations

import torch

GEMMA_REPO = "Efficient-Large-Model/gemma-2-2b-it"
MODEL_MAX_LENGTH = 300
NEGATIVE_PROMPT = "low quality, worst quality, over-saturated, blurry, deformed, watermark"

CHI_PROMPT = [
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]


class GemmaEncoder:
    """Frozen caption encoder."""

    def __init__(self, repo=GEMMA_REPO, device="cuda", max_length=MODEL_MAX_LENGTH, dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.repo, self.device, self.max_length = repo, device, int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype).get_decoder().to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.chi = "\n".join(CHI_PROMPT)
        self.padded_length = len(self.tokenizer.encode(self.chi)) + self.max_length - 2
        self.select_index = [0] + list(range(-self.max_length + 1, 0))

    @torch.no_grad()
    def encode(self, captions):
        """list[str] -> (emb (B, 300, 2304) float32, mask (B, 300) long)."""
        prompts = [self.chi + c for c in captions]
        tok = self.tokenizer(prompts, padding="max_length", max_length=self.padded_length,
                             truncation=True, return_tensors="pt").to(self.device)
        h = self.model(tok.input_ids, attention_mask=tok.attention_mask)[0]
        emb = h[:, self.select_index, :].float()
        mask = tok.attention_mask[:, self.select_index]
        return emb.cpu(), mask.cpu()

    @torch.no_grad()
    def null_embedding(self, prompt=""):
        """The CFG / caption-dropout embedding, (1, 300, 2304): `prompt` without the CHI prefix, padded
        to 300 tokens, all hidden states kept. Training uses ""; evaluation the negative prompt."""
        tok = self.tokenizer([prompt], padding="max_length", max_length=self.max_length, truncation=True,
                             return_tensors="pt").to(self.device)
        h = self.model(tok.input_ids, attention_mask=tok.attention_mask)[0]
        return h.float().cpu()
