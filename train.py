"""Supervised LoRA/full fine-tuning; one CPU or one CUDA GPU."""
import argparse
import json
import math
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup, set_seed

from data_utils import read_rows, validate_splits
from harrier_classifier import Collator, new_model, save_bundle, tokenizer_from, load_bundle


def evaluate(model, loader, device, labels):
    model.eval()
    truth, predicted, loss_sum = [], [], 0.0
    with torch.inference_mode():
        for inputs, targets in loader:
            targets = targets.to(device)
            logits = model(**{k: v.to(device) for k, v in inputs.items()})
            loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
            truth.extend(targets.cpu().tolist())
            predicted.extend(logits.argmax(-1).cpu().tolist())
    ids = list(range(len(labels)))
    return {"loss": loss_sum / len(truth), "accuracy": accuracy_score(truth, predicted),
            "macro_f1": f1_score(truth, predicted, labels=ids, average="macro", zero_division=0),
            "weighted_f1": f1_score(truth, predicted, labels=ids, average="weighted", zero_division=0),
            "report": classification_report(truth, predicted, labels=ids, target_names=labels,
                                            output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=ids).tolist()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="./harrier-oss-v1-0.6b")
    p.add_argument("--train", default="data/train.jsonl")
    p.add_argument("--valid")
    p.add_argument("--test")
    p.add_argument("--output", default="outputs/harrier_classifier")
    p.add_argument("--mode", choices=["lora", "full"], default="lora")
    p.add_argument("--instruction", default="Classify the text into the appropriate category.")
    p.add_argument("--max-length", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    args = p.parse_args()
    if min(args.batch_size, args.grad_accum, args.epochs, args.max_length, args.lora_r, args.patience) < 1:
        p.error("batch-size / grad-accum / epochs / max-length / lora-r / patience 必须大于零")
    if not 0 < args.val_ratio < 1 or not 0 <= args.warmup_ratio <= 1 or not 0 <= args.dropout < 1:
        p.error("val-ratio、warmup-ratio 或 dropout 不合法")
    if args.head_lr <= 0 or (args.lr is not None and args.lr <= 0) or args.weight_decay < 0:
        p.error("学习率必须大于零，weight-decay 不能为负数")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录非空，请选择新目录以免覆盖已有训练")
    rows = read_rows(args.train)
    if args.valid:
        train, valid = rows, read_rows(args.valid)
    else:
        try:
            train, valid = train_test_split(rows, test_size=args.val_ratio, random_state=args.seed,
                                           stratify=[r["label"] for r in rows])
        except ValueError as exc:
            raise ValueError("无法分层划分：增加每类样本数、调整 val-ratio 或提供 --valid") from exc
    test = read_rows(args.test) if args.test else None
    labels = validate_splits(train, valid, test)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    precision = args.precision
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "fp32"
    if device.type == "cpu" and precision != "fp32":
        raise ValueError("CPU 模式请使用 --precision fp32")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("当前 GPU 不支持 bf16，请改为 fp16 或 fp32")
    tokenizer = tokenizer_from(args.model)
    collator = Collator(tokenizer, args.max_length, args.instruction, {v: i for i, v in enumerate(labels)})
    def loader(data, shuffle=False):
        return DataLoader(data, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=collator, num_workers=0, pin_memory=device.type == "cuda")
    train_loader, valid_loader = loader(train, True), loader(valid)
    model = new_model(args, len(labels)).to(device)
    lr = args.lr if args.lr is not None else (2e-4 if args.mode == "lora" else 2e-5)
    groups = []
    for is_head in (False, True):
        for decay in (False, True):
            params = [p for n, p in model.named_parameters() if p.requires_grad
                      and n.startswith("classifier.") == is_head and (p.ndim >= 2) == decay]
            if params:
                groups.append({"params": params, "lr": args.head_lr if is_head else lr,
                               "weight_decay": args.weight_decay if decay else 0.0})
    optimizer = torch.optim.AdamW(groups)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    weights = None
    if args.class_weights:
        counts = torch.bincount(torch.tensor([labels.index(r["label"]) for r in train]), minlength=len(labels))
        weights = (len(train) / (len(labels) * counts.float())).to(device)
    output.mkdir(parents=True, exist_ok=True)
    base_model = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    metadata = {"base_model": base_model, "mode": args.mode, "labels": labels,
                "max_length": args.max_length, "instruction": args.instruction, "dropout": args.dropout}
    (output / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    # Store actual split membership for reproducibility, including auto-split results.
    for name, data in [("train", train), ("valid", valid), ("test", test)]:
        if data is not None:
            (output / f"{name}_split.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8")
    print(f"device={device}, precision={precision}, labels={labels}")
    print(f"train={len(train)}, valid={len(valid)}, trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    best, stale, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, (inputs, targets) in enumerate(progress):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = targets.to(device)
            # Normalize by samples in this accumulation window, including its short final batch.
            window_start = (step // args.grad_accum) * args.grad_accum * args.batch_size
            window_samples = min(args.grad_accum * args.batch_size, len(train) - window_start)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=precision != "fp32"):
                print("input_ids shape:", tuple(inputs["input_ids"].shape), flush=True)
                logits = model(**inputs)
                per_sample = F.cross_entropy(logits.float(), targets, weight=weights, reduction="none")
                loss = per_sample.sum() / window_samples
            if not torch.isfinite(loss):
                raise FloatingPointError("训练 loss 非有限值；请降低学习率或改用 fp32")
            scaler.scale(loss).backward()
            loss_sum += per_sample.detach().sum().item()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{per_sample.mean().item():.4f}")
        metrics = evaluate(model, valid_loader, device, labels)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train), "valid": metrics})
        (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"valid loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
        if metrics["macro_f1"] > best:
            best, stale = metrics["macro_f1"], 0
            save_bundle(model, tokenizer, output / "best", {**metadata, "best_epoch": epoch})
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping")
                break
    if test is not None:
        # Release optimizer state and the old model before loading the best model.
        del optimizer, scheduler, groups, params, model, logits, loss, per_sample
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model, _, _ = load_bundle(output / "best", device)
        metrics = evaluate(model, loader(test), device, labels)
        (output / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"test accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    print(f"Best model: {output / 'best'} (valid macro_f1={best:.4f})")


if __name__ == "__main__":
    main()
