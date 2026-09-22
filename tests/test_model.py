"""Small random Qwen3: no model download; requires training dependencies."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen3Config, Qwen3Model, AutoModel
from peft import PeftModel
from safetensors.torch import load_file, save_file

from harrier_classifier import HarrierClassifier, last_token_pool, new_model


class ModelTests(unittest.TestCase):
    def test_pooling(self):
        hidden = torch.arange(24).reshape(2, 4, 3).float()
        mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
        torch.testing.assert_close(last_token_pool(hidden, mask), torch.stack([hidden[0, 1], hidden[1, 3]]))

    def test_lora_and_full_update_and_reload(self):
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8, max_position_embeddings=64)
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base"
            Qwen3Model(config).save_pretrained(base)
            for mode in ["lora", "full"]:
                args = SimpleNamespace(model=str(base), max_length=16, mode=mode,
                                       lora_r=2, gradient_checkpointing=True, dropout=0.0)
                model = new_model(args, 2).train()
                inputs = {"input_ids": torch.tensor([[1, 2, 3], [2, 3, 4]]),
                          "attention_mask": torch.ones(2, 3, dtype=torch.long)}
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
                torch.nn.functional.cross_entropy(model(**inputs), torch.tensor([0, 1])).backward()
                optimizer.step()
                changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
                self.assertTrue(any(n.startswith("classifier") for n in changed))
                self.assertTrue(any(n.startswith("backbone") for n in changed))
                if mode == "lora":
                    self.assertTrue(all("lora_" in n or n.startswith("classifier") for n in changed))
                checkpoint = Path(d) / mode
                model.backbone.save_pretrained(checkpoint)
                save_file(model.classifier.state_dict(), str(Path(d) / "head.safetensors"))
                backbone = (PeftModel.from_pretrained(AutoModel.from_pretrained(base), checkpoint)
                            if mode == "lora" else AutoModel.from_pretrained(checkpoint))
                restored = HarrierClassifier(backbone, 2, dropout=0.0).eval()
                restored.classifier.load_state_dict(load_file(str(Path(d) / "head.safetensors")))
                model.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(**inputs), restored(**inputs))
                    if mode == "lora":
                        base_embeddings = restored.encode_base(**inputs)
                        self.assertEqual(base_embeddings.shape, (2, config.hidden_size))
                        torch.testing.assert_close(base_embeddings.norm(dim=-1), torch.ones(2))
                    else:
                        with self.assertRaises(RuntimeError):
                            restored.encode_base(**inputs)


if __name__ == "__main__":
    unittest.main()
