# Harrier 文本分类后训练

基于项目内 `harrier-oss-v1-0.6b`，支持中文、英文等文本的**单标签分类**（二分类、多分类）。每条文本只对应一个标签；多标签任务需要修改损失函数与预测逻辑。

模型流程：任务指令 + 文本 → Harrier / Qwen3 → 最后一个有效 token → L2 归一化 → Dropout → 线性分类头 → 交叉熵。

默认 LoRA 微调注意力和 MLP 投影，同时训练分类头；`--mode full` 更新全部模型参数。训练直接调用可求导的模型 forward。分类结构保留了[官方模型卡](https://huggingface.co/microsoft/harrier-oss-v1-0.6b)的池化方式；LoRA 使用 [PEFT](https://huggingface.co/docs/peft/package_reference/lora)。

## 1. 环境

建议 Python 3.10–3.12，单张 NVIDIA GPU。先按 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)安装适合驱动的 CUDA 版 PyTorch，再安装其余依赖：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

代码支持 CPU，但完整 0.6B 训练较慢。默认使用 FP32 主权重和 GPU 混合精度，支持梯度检查点与梯度累积；不是量化训练，显存占用取决于文本长度和批大小。本实现为单进程、单卡训练，不使用 torchrun。

## 2. 准备标注数据

支持 UTF-8 JSONL（每行一个对象）或 CSV（包含 `text,label` 表头）。标签可用字符串或整数，内部统一成字符串：

```json
{"text":"球队赢得了决赛冠军","label":"体育"}
{"text":"新款手机采用了新的芯片","label":"科技"}
{"text":"今天学习如何制作蛋糕","label":"美食"}
```

训练集至少包含两类。默认按类别分层划分 20% 验证集，需要每类有足够样本；也可自行提供 `--valid`。未知标签、冲突标签、跨集合重复文本会报错；集合内部同标签重复文本会去重。同一用户、文档或同源改写的样本应在数据准备阶段按组划分，程序只能检测完全相同的文本。

`examples/train.jsonl` 是 18 条演示数据，仅用于检查流程，不能用于判断实际分类效果。

## 3. 开始训练

先用示例检查流程（Windows、Linux 均可执行以下单行命令）：

```bash
python train.py --model ./harrier-oss-v1-0.6b --train examples/train.jsonl --output outputs/demo --epochs 1 --max-length 128 --batch-size 1 --grad-accum 4
```

使用真实标注数据进行 LoRA 后训练：

```bash
python train.py --model ./harrier-oss-v1-0.6b --train data/train.jsonl --valid data/valid.jsonl --test data/test.jsonl --output outputs/lora_v1 --mode lora --epochs 5 --batch-size 4 --grad-accum 8 --max-length 512 --lr 2e-4 --head-lr 1e-3 --instruction "根据文本内容判断其所属类别。"
```

`data/` 下文件需自行准备。没有独立测试集时省略 `--test`；没有验证集文件时省略 `--valid`。也可将模型路径换成 `/mnt/harrier/harrier-oss-v1-0.6b` 或在线仓库 `microsoft/harrier-oss-v1-0.6b`。

全参数微调：

```bash
python train.py --train data/train.jsonl --valid data/valid.jsonl --output outputs/full_v1 --mode full --lr 2e-5 --head-lr 1e-3 --batch-size 2 --grad-accum 16
```

常用参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--mode` | `lora` | `lora` / `full` |
| `--max-length` | 512 | 指令和文本合计 token 上限，超长从尾部截断 |
| `--batch-size` | 4 | 每个微批样本数 |
| `--grad-accum` | 8 | 梯度累积次数，完整窗口有效 batch=32 |
| `--lr` | LoRA 2e-4 / full 2e-5 | 编码器学习率 |
| `--head-lr` | 1e-3 | 新分类头学习率 |
| `--lora-r` | 16 | LoRA 秩，alpha=2r |
| `--precision` | auto | CUDA 优先 bf16，否则 fp16；CPU fp32 |
| `--class-weights` | 关闭 | 启用训练集逆频率类别权重 |
| `--patience` | 3 | 验证 macro-F1 连续未提升的 epoch 数 |
| `--seed` | 42 | 随机种子，不保证跨设备逐位一致 |

显存不足时先减小 `--batch-size` 和 `--max-length`，可增加 `--grad-accum` 保持有效批大小。默认启用梯度检查点，使用 `--no-gradient-checkpointing` 关闭。训练与预测自动使用相同指令；指令无需包含答案标签。

按验证集 macro-F1 保存最佳模型，并在训练结束后用最佳模型评估独立测试集。输出包含 accuracy、macro-F1、weighted-F1、逐类 precision/recall/F1 和混淆矩阵（行是真实标签，列是预测标签，顺序见 `labels`）。启用类别权重时，训练 loss 是加权均值，验证 loss 是普通交叉熵，不能直接比较。

## 4. 预测

```bash
python predict.py --checkpoint outputs/lora_v1/best --text "这支球队赢得了冠军" "新芯片性能提升明显"
python predict.py --checkpoint outputs/lora_v1/best --input data/predict.jsonl --output outputs/predictions.jsonl
```

预测文件只需 `text` 字段。每条结果同时输出 LoRA 分类结果（`label`、`score`、`probabilities`）和原始 Harrier 向量（`embedding`）。原始向量使用不带分类指令的文本，并在临时关闭 LoRA adapter 后计算；它经过末位 token 池化和 L2 归一化，维度为 1024。LoRA 调整后的中间向量只在模型内部用于分类，不对外返回。一次预测需要执行两次 backbone forward，因此该预测接口只接受 LoRA checkpoint。分类概率未做校准。移动 LoRA 模型到另一台机器后，使用 `--base-model /new/path/harrier-oss-v1-0.6b` 指定原始基座路径；基座必须与训练时一致。

## 5. 保存内容与验证

```text
outputs/lora_v1/
  run_config.json           # 命令参数
  train_split.jsonl         # 实际训练数据（去重后）
  valid_split.jsonl         # 实际验证数据
  test_split.jsonl          # 提供 --test 时保存
  history.json              # 每轮损失与完整验证指标
  test_metrics.json         # 提供 --test 时生成
  best/
    classifier_config.json # 标签顺序、任务指令、基座路径等
    classifier.safetensors # 分类头
    tokenizer/             # 分词器
    backbone/              # LoRA adapter 或完整微调基座
```

LoRA 保存内容需要原始基座才能预测；full 模式包含完整编码器。使用 `predict.py` 加载分类模型。未保存 optimizer/scheduler 状态，因此不支持精确断点续训；输出目录非空时会拒绝覆盖。

```bash
python -m unittest discover -s tests -p test_data_utils.py -v
python -m unittest discover -s tests -p test_model.py -v
```

模型测试使用随机初始化的微型 Qwen3，检查池化、LoRA/全参数反向传播和保存重载，不下载真实模型。代码交付时所在环境缺少 PyTorch/Transformers/PEFT，只执行了语法与数据处理测试；模型测试和真实数据训练尚未执行。
