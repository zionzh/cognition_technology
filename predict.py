"""Predict using a saved classifier (labels and instruction restored automatically)."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_utils import read_rows
from harrier_classifier import Collator, load_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="outputs/harrier_classifier/best")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--text", nargs="+", help="Preformatted text; for articles use --input with title/content")
    source.add_argument("--input",  default="data/test.jsonl", help="JSONL/CSV containing title/content (legacy text also supported)")
    p.add_argument("--output",  default="outputs/test_predictions.jsonl", help="Optional JSONL output")
    p.add_argument("--base-model", help="Override original model location for a LoRA checkpoint")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch-size 必须大于零")
    rows = read_rows(args.input, labeled=False) if args.input else [{"text": t.strip()} for t in args.text]
    if any(not r["text"] for r in rows):
        p.error("text 不能为空")
    if args.output and Path(args.output).exists():
        p.error("输出文件已存在，请选择新文件名")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model, tokenizer, config = load_bundle(args.checkpoint, device, args.base_model)
    if config["mode"] != "lora":
        p.error("该预测接口需要 LoRA checkpoint，以便同时输出原始 Harrier embedding")
    adapted_collator = Collator(tokenizer, config["max_length"], config["instruction"])
    base_collator = Collator(tokenizer, config["max_length"], instruction=None)

    def collate(batch):
        adapted_inputs, _ = adapted_collator(batch)
        base_inputs, _ = base_collator(batch)
        return adapted_inputs, base_inputs

    loader = DataLoader(rows, batch_size=args.batch_size, collate_fn=collate)
    results, offset = [], 0
    with torch.inference_mode():
        for inputs, base_inputs in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs)
            base_inputs = {k: v.to(device) for k, v in base_inputs.items()}
            base_embeddings = model.encode_base(**base_inputs)
            probs = logits.float().softmax(-1).cpu()
            base_embeddings = base_embeddings.cpu()
            for row_index, prob in enumerate(probs):
                index = prob.argmax().item()
                item = {"text": rows[offset]["text"], "label": config["labels"][index],
                        "score": prob[index].item(),
                        "probabilities": dict(zip(config["labels"], prob.tolist())),
                        "embedding": base_embeddings[row_index].tolist()}
                results.append(json.dumps(item, ensure_ascii=False))
                offset += 1
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("\n".join(results) + "\n", encoding="utf-8")
    else:
        print("\n".join(results))


if __name__ == "__main__":
    main()
