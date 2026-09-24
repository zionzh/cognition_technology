"""Single-label Harrier classifier; shared training/inference implementation."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file, save_file
from data_utils import normalize_text


def last_token_pool(hidden, mask):
    # Works for left/right padding, including EOS == PAD: use mask, not token IDs.
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = positions.masked_fill(mask == 0, -1).max(dim=1).values
    if (indices < 0).any():
        raise ValueError("Encountered an empty token sequence")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]


class HarrierClassifier(nn.Module):
    def __init__(self, backbone, num_labels, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(backbone.config.hidden_size, num_labels)

    def _encode_backbone(self, **inputs):
        outputs = self.backbone(**inputs, use_cache=False, return_dict=True)
        embedding = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        return F.normalize(embedding.float(), p=2, dim=-1)

    def _encode_adapted(self, **inputs):
        """Internal LoRA-adapted representation used only for classification."""
        return self._encode_backbone(**inputs)

    def encode_base(self, **inputs):
        """Return the frozen base Harrier embedding with LoRA disabled."""
        if not isinstance(self.backbone, PeftModel):
            raise RuntimeError("原始 embedding 只可从 LoRA 模型获取；full 模式已经修改基座权重")
        # PEFT keeps the base weights in the same object. This context manager
        # bypasses every adapter without creating a second 0.6B model copy.
        with self.backbone.disable_adapter():
            return self._encode_backbone(**inputs)

    def forward(self, **inputs):
        adapted_embedding = self._encode_adapted(**inputs)
        logits = self.classifier(self.dropout(adapted_embedding))
        return logits


class Collator:
    def __init__(self, tokenizer, max_length, instruction, label2id=None):
        self.tokenizer, self.max_length = tokenizer, max_length
        self.instruction, self.label2id = instruction, label2id

    def __call__(self, rows):
        texts = [normalize_text(r["text"]) for r in rows]
        if self.instruction:
            texts = [f'Instruct: {self.instruction}\nQuery: {text}' for text in texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt",
                                return_token_type_ids=False)
        labels = None if self.label2id is None else torch.tensor(
            [self.label2id[r["label"]] for r in rows], dtype=torch.long)
        return inputs, labels


def tokenizer_from(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have a PAD or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def new_model(args, num_labels):
    # FP32 master weights + autocast; avoids FP16 optimizer underflow.
    backbone = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32)
    if args.max_length > backbone.config.max_position_embeddings:
        raise ValueError("max-length 超过模型上下文长度")
    backbone.config.use_cache = False
    if args.mode == "lora":
        backbone = get_peft_model(backbone, LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=args.lora_r,
            lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
    if args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return HarrierClassifier(backbone, num_labels, args.dropout)


def save_bundle(model, tokenizer, directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(directory / "backbone", safe_serialization=True)
    tokenizer.save_pretrained(directory / "tokenizer")
    save_file({k: v.detach().cpu().contiguous() for k, v in model.classifier.state_dict().items()},
              str(directory / "classifier.safetensors"))
    (directory / "classifier_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_bundle(directory, device, base_model=None):
    directory = Path(directory)
    config = json.loads((directory / "classifier_config.json").read_text(encoding="utf-8"))
    if config["mode"] == "lora":
        backbone = AutoModel.from_pretrained(base_model or config["base_model"], torch_dtype=torch.float32)
        backbone = PeftModel.from_pretrained(backbone, directory / "backbone")
    else:
        backbone = AutoModel.from_pretrained(directory / "backbone", torch_dtype=torch.float32)
    model = HarrierClassifier(backbone, len(config["labels"]), config["dropout"])
    model.classifier.load_state_dict(load_file(str(directory / "classifier.safetensors")))
    model.to(device).eval()
    return model, tokenizer_from(directory / "tokenizer"), config
