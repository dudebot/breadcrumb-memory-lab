import importlib.util
import unittest

import torch


@unittest.skipUnless(importlib.util.find_spec("transformers"), "optional language-model dependencies not installed")
class LanguageMemoryTests(unittest.TestCase):
    def test_gradients_cross_frozen_backbone_into_compressor(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from breadcrumb_memory.lm_memory import SoftMemory, student_logits
        torch.manual_seed(3)
        torch.set_num_threads(2)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32,
            intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2)).eval().requires_grad_(False)
        writer = SoftMemory(32, slots=4, bottleneck=16)
        old = torch.randn(2, 12, 32, requires_grad=True)
        memory = writer(old, torch.ones(2, 12, dtype=torch.long))
        local = {"input_ids": torch.randint(64, (2, 8)), "attention_mask": torch.ones(2, 8, dtype=torch.long)}
        loss = torch.nn.functional.cross_entropy(student_logits(model, memory, local), torch.tensor([3, 11]))
        loss.backward()
        self.assertIsNone(old.grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertGreater(writer.output_projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(writer.queries.grad.abs().sum().item(), 0)

    def test_masked_padding_cannot_change_memory(self):
        from breadcrumb_memory.lm_memory import SoftMemory
        torch.manual_seed(4)
        writer = SoftMemory(32, slots=4, bottleneck=16).eval()
        original = torch.randn(2, 5, 32)
        extended = torch.cat((torch.randn(2, 3, 32)*1000, original), dim=1)
        plain = writer(original, torch.ones(2, 5, dtype=torch.long))
        padded = writer(extended, torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]]).repeat(2, 1))
        torch.testing.assert_close(plain, padded)


if __name__ == "__main__":
    unittest.main()
