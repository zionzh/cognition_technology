# from transformers import AutoTokenizer, AutoModel

# path = r"/mnt/harrier/harrier-oss-v1-0.6b"

# tokenizer = AutoTokenizer.from_pretrained(
#     path,
#     local_files_only=True
# )

# model = AutoModel.from_pretrained(
#     path,
#     local_files_only=True
# )

from sentence_transformers import SentenceTransformer

model_path = "/mnt/harrier/harrier-oss-v1-0.6b"

model = SentenceTransformer(
    model_path,
    local_files_only=True
)

texts = [
    """4月7日，微软必应（Bing）团队正式推出了名为“Harrier”的全新词嵌入模型系列，标志着全球搜索、检索及人工智能代理的底层逻辑将迎来重塑。Harrier系列包含三个不同规格的版本，其中旗舰级27B模型在最新的多语言MTEBv2基准测试中，表现出色，超越了OpenAI、亚马逊及Google Gemini等主流专有模型，荣登榜首。

Harrier模型的技术底座展现了极高的工业水准，其支持超过100种语言，且上下文窗口高达32,000个词元。这意味着Harrier在处理复杂语境和长文本时，具备显著优势。微软的训练策略同样值得关注，团队不仅使用了超过20亿个真实示例，还引入了来自GPT-5的合成数据进行强化。这种高质量数据的结合，使得Harrier在语义理解和信息检索上达到新的高度。

值得一提的是，除了旗舰级的270亿参数版本，微软还推出了0.6B和2.7B的小参数版本，以适应不同算力环境。这些模型均通过MIT许可证在HuggingFace平台上开放，方便开发者使用。嵌入模型作为AI系统中信息组织与检索的关键技术，其性能直接决定了RAG（检索增强生成）系统的准确性。

微软计划将Harrier技术深度集成至Bing搜索引擎及新型AI代理服务中，随着人工智能逐步迈向多步骤任务的自主化，Harrier的开源不仅为开发者提供了可替代专有模型的高性能工具，更是开源生态在语义表示能力上完成对顶尖闭源方案的阶段性跨越。这一进展，预计将加速AI代理在全球多语言环境下的落地进程。

在当今信息时代，AI的快速发展已成为社会各领域变革的驱动力。Harrier的推出无疑将引发更多开发者的关注与参与，促进技术的创新与应用。同时，这也为全球用户提供了更为丰富和智能的搜索体验。随着Harrier的开源，未来的AI技术将更加开放、共享，推动整个行业向前迈进。""",
    "iPhone 电池维修方法",
    "今天北京天气很好",
]

embeddings = model.encode(texts)

print(embeddings.shape)
print(model.device)