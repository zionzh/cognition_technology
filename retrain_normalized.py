"""Run in the IDE: prepare normalized data, train a fresh LoRA, then test it.

--prepare-only performs the dependency-free data preparation without training.
Original datasets and previous training runs are never overwritten.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from data_preparation import prepare_data

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(ROOT / "data/train.jsonl"))
    parser.add_argument("--test", default=str(ROOT / "data/test.jsonl"))
    parser.add_argument("--model", default=str(ROOT / "harrier-oss-v1-0.6b"))
    parser.add_argument("--output", help="New experiment directory; default includes timestamp")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=.2)
    args = parser.parse_args()
    if min(args.max_length, args.batch_size, args.grad_accum, args.epochs) < 1:
        parser.error("长度、批大小、梯度累积次数、训练轮数必须大于零")
    # This entry point always starts from the original base, never from a saved adapter.
    if (Path(args.model) / "adapter_config.json").exists() or (Path(args.model) / "classifier_config.json").exists():
        parser.error("--model 必须指定原始 Harrier 基座，不可指定之前的分类 checkpoint")
    output = (Path(args.output).resolve() if args.output else
              ROOT / "outputs" / ("normalized_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    if output.exists():
        parser.error("实验目录已存在，请选择新目录")
    report = prepare_data(args.train, args.test, output / "data", args.val_ratio, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    command = [sys.executable, str(ROOT / "train.py"),
               "--model", args.model, "--train", str(output / "data/train.jsonl"),
               "--valid", str(output / "data/valid.jsonl"),
               "--test", str(output / "data/test.jsonl"), "--output", str(output / "training"),
               "--mode", "lora", "--max-length", str(args.max_length),
               "--batch-size", str(args.batch_size), "--grad-accum", str(args.grad_accum),
               "--epochs", str(args.epochs), "--lr", "2e-4", "--head-lr", "1e-3",
               "--seed", str(args.seed), "--val-ratio", str(args.val_ratio), "--patience", "3"]
    (output / "training_command.json").write_text(
        json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Experiment: {output}", flush=True)
    if args.prepare_only:
        print("Data preparation finished. Training has not been run.")
        return
    # Dependency check before a potentially long training run; GPU selection stays with train.py.
    try:
        import torch
        import transformers
        import peft
        import sklearn
    except ImportError as exc:
        raise SystemExit(f"数据已准备完成，但当前 Python 缺少训练依赖：{exc}。请在原训练环境运行此脚本。") from exc
    subprocess.run(command, cwd=ROOT, check=True)
    metrics_path = output / "training/test_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(f"Test accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    print(f"Prediction counts: {metrics['prediction_counts']}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
