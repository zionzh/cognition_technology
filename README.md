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

支持 UTF-8 JSONL（每行一个对象）或 CSV（包含 `title,content,label` 表头）。训练、验证、测试文件采用相同格式。标签可用字符串或整数，内部统一成字符串：

```json
{"id":"1","title":"球队夺冠","content":"球队赢得了决赛冠军。","label":"体育"}
{"id":"2","title":"新款手机发布","content":"新款手机采用了新的芯片。","label":"科技"}
{"id":"3","title":"蛋糕制作教程","content":"今天学习如何制作蛋糕。","label":"美食"}
```

读取时只使用标题和正文构造模型的 `text`，`id` 和 `label` 不会拼入文本。例如第一条转换为：

```text
title: 球队夺冠
content: 球队赢得了决赛冠军。
```

标题和正文去除首尾空白，统一 CRLF/CR 为 LF，并删除空白行（包括只有空格、制表符的行），所以连续换行统一为一个换行。该规则同时适用于训练、验证、测试、文件预测和 `--text` 预测，避免模型利用段落空行数判断标签。某字段缺失、为 `null` 或只有空白时，跳过该字段及其提示，两个字段都为空则报错。非空字段必须是字符串。若数据同时带有旧 `text` 字段，优先使用 `title/content`，不会用旧 `text` 替代空标题和空正文。完全不含 `title/content` 的数据仍兼容原来的 `text,label` 格式，以便读取旧数据、示例和保存的划分文件。

分类分支在拼接文本前添加任务指令；原始 Harrier embedding 分支使用同一份 `title: ...\ncontent: ...` 文本，但不添加分类指令。分词阶段继续按 `--max-length` 截断，训练入口默认 5000 tokens；分类分支的上限包含指令和字段提示。

训练集至少包含两类。`train.py` 默认按类别分层划分 20% 验证集，也可提供 `--valid`；它只检查完全相同的文本。针对当前真实数据，请使用 `retrain_normalized.py`，先按下述文章分组规则清洗并生成独立的 train/valid/test，再交给训练程序。

`examples/train.jsonl` 是 18 条演示数据，仅用于检查流程，不能用于判断实际分类效果。

## 3. 开始训练

针对空行格式问题重新训练：在已安装依赖的训练环境中，直接 Run `retrain_normalized.py`，或执行：

```bash
python retrain_normalized.py
```

该入口使用项目根目录下的原始 `harrier-oss-v1-0.6b`，重新创建 LoRA 和分类头，使用 `data/train.jsonl` 和 `data/test.jsonl`。保留原始数据、原始标签及旧实验，生成新的 `outputs/normalized_时间戳/`。当前推荐直接 Run 此文件，不要直接 Run `train.py` 绕过文章分组清洗。

数据准备规则（`data_preparation.py`，不依赖模型库）：

- 统一换行。另为重复检测构造忽略空白、标点、大小写及兼容字符差异的比较文本；这些额外变换不会用于模型输入。
- 完全相同文本、至少 12 个字母/数字的相同标题，或正文高度重叠的记录合并为文章组。正文至少 200 字符、100 个不同的五字符片段；片段包含率达到 90%，或 Jaccard 达到 80%，即连接两篇文章。通过传递连接形成完整分组。
- 同组标签冲突时，将训练记录隔离到审计文件，不猜测或改写正确标签。同组标签一致时，保留最长版本，避免重复训练；每个文章组只保留一个代表，因此不会跨训练/验证集。
- 与测试集同组的训练记录全部排除。较弱的正文重叠（包含率至少 65%）也会列入复核清单；其中涉及测试集的训练文章组同样暂时排除，以保护测试独立性。其他较弱匹配只提示复核，不自动合并。
- 按类别、固定种子 42 划分约 80% 训练、20% 验证，每类至少各一条；通过显式 `--valid` 传入训练程序。清洗后每类不足两篇独立文章会报错。
- 测试数据保留原始标签，发现的测试标签冲突会记入报告。它仍是参考评估集，需人工确认标注标准；这些规则不能识别所有语义改写或同事件报道。

训练结束后自动加载验证 macro-F1 最佳模型，保存验证集和测试集逐条预测及测试汇总指标。主要训练参数保持不变，方便先观察数据调整的影响：max-length=5000、batch-size=1、grad-accum=32、epochs=5、LoRA lr=2e-4、head lr=1e-3、patience=3。可调整长度、批大小、累积次数、轮数、`--seed` 和 `--val-ratio`。

```text
outputs/normalized_时间戳/
  data/train.jsonl             # 清洗并完成划分后的实际训练集
  data/valid.jsonl             # 按独立文章划分的验证集
  data/test.jsonl              # 规范化的测试数据，标签不变
  data/preparation_report.json # 来源文件哈希、分组规则、排除数量、类别数量
  data/audit.jsonl             # 全部原始记录及行号、文章组、处理方式、最终集合
  data/label_conflicts.jsonl   # 同组标签冲突的记录，等待人工复核
  data/duplicate_matches.jsonl # 自动分组的匹配依据
  data/similarity_review.jsonl # 较弱的相似匹配；a/b 对应 audit 的 record_index
  training_command.json       # 此次训练命令参数
  training/best/              # 新分类模型；包含 text_normalization 版本
  training/history.json       # 训练与验证记录
  training/test_metrics.json  # 测试准确率、F1、混淆矩阵及预测类别计数
  training/test_predictions.jsonl  # 最佳模型的真实标签、预测标签、概率、correct
  training/valid_predictions.jsonl # 最佳模型的逐条验证预测
```

只准备数据而不训练，可运行 `python retrain_normalized.py --prepare-only`；这一步不依赖 PyTorch。正式训练仍需在有训练依赖的环境运行。预测新模型时，通过 `--checkpoint outputs/normalized_时间戳/training/best` 指定它，并使用新输出文件名。旧模型没有学习新的数据格式，仅清洗预测输入并不能替代重新训练。规范化只消除已发现的空行特征，不保证新的测试准确率，仍需核对标注与数据来源差异。

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
| `--max-length` | 5000 | 指令和文本合计 token 上限，超长从尾部截断 |
| `--batch-size` | 1 | 每个微批样本数 |
| `--grad-accum` | 32 | 梯度累积次数，完整窗口有效 batch=32 |
| `--lr` | LoRA 2e-4 / full 2e-5 | 编码器学习率 |
| `--head-lr` | 1e-3 | 新分类头学习率 |
| `--lora-r` | 16 | LoRA 秩，alpha=2r |
| `--precision` | auto | CUDA 优先 bf16，否则 fp16；CPU fp32 |
| `--class-weights` | 关闭 | 启用训练集逆频率类别权重 |
| `--patience` | 3 | 验证 macro-F1 连续未提升的 epoch 数 |
| `--seed` | 42 | 随机种子，不保证跨设备逐位一致 |

显存不足时先减小 `--batch-size` 和 `--max-length`，可增加 `--grad-accum` 保持有效批大小。默认启用梯度检查点，使用 `--no-gradient-checkpointing` 关闭。训练与预测自动使用相同指令；指令无需包含答案标签。

按验证集 macro-F1 保存最佳模型，并在训练结束后用最佳模型评估独立测试集。输出包含 accuracy、macro-F1、weighted-F1、逐类 precision/recall/F1 和混淆矩阵（行是真实标签，列是预测标签，顺序见 `labels`）。启用类别权重时，训练 loss 是加权均值，验证 loss 是普通交叉熵，不能直接比较。

提供 `--test` 时，还会保存最佳模型的 `test_predictions.jsonl` 和 `valid_predictions.jsonl`：`true_label` 为数据标签，`predicted_label` 为预测，`correct=false` 即错分，`split_row` 对应实际划分文件中的行号。这些评估文件不重复计算 embedding；需要原始 Harrier embedding 时仍使用 `predict.py`，其接口保持不变。

## 4. 预测

```bash
python predict.py --checkpoint outputs/lora_v1/best --text "这支球队赢得了冠军" "新芯片性能提升明显"
python predict.py --checkpoint outputs/lora_v1/best --input data/predict.jsonl --output outputs/predictions.jsonl
```

预测文件使用与训练相同的 `title/content` 字段，无需 `label`，也兼容旧 `text` 格式。新闻数据建议通过 `--input` 传入文件；`--text` 直接接收已拼接好的文本，不会自动添加字段提示。每条结果的 `text` 是拼接后的文本，同时输出 LoRA 分类结果（`label`、`score`、`probabilities`）和原始 Harrier 向量（`embedding`）。原始向量使用不带分类指令的拼接文本，并在临时关闭 LoRA adapter 后计算；它经过末位 token 池化和 L2 归一化，维度为 1024。LoRA 调整后的中间向量只在模型内部用于分类，不对外返回。一次预测需要执行两次 backbone forward，因此该预测接口只接受 LoRA checkpoint。分类概率未做校准。移动 LoRA 模型到另一台机器后，使用 `--base-model /new/path/harrier-oss-v1-0.6b` 指定原始基座路径；基座必须与训练时一致。

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
python -m unittest discover -s tests -p test_data_preparation.py -v
python -m unittest discover -s tests -p test_model.py -v
```

模型测试使用随机初始化的微型 Qwen3，检查池化、LoRA/全参数反向传播和保存重载，不下载真实模型。代码交付时所在环境缺少 PyTorch/Transformers/PEFT，只执行了语法与数据处理测试；模型测试和真实数据训练尚未执行。
