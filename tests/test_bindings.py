import importlib.util
import random
import unittest

import torch


@unittest.skipUnless(importlib.util.find_spec("transformers"), "optional LM dependencies not installed")
class BindingTests(unittest.TestCase):
    def test_counterfactual_changes_only_old_assignments(self):
        from breadcrumb_memory.binding_eval import quartets, texts
        cases = quartets(random.Random(2), 8)
        for example in cases:
            a, b = example["original"], example["swapped"]
            self.assertEqual(a[2:], b[2:])
            self.assertEqual(sorted(x[1] for x in a[:2]), sorted(x[1] for x in b[:2]))
            self.assertNotEqual(a[0][1], b[0][1])
            self.assertEqual([x[0] for x in a], [x[0] for x in b])
        t = texts(cases)
        self.assertEqual(len(t["old"]), 2*len(cases))
        self.assertEqual(len(t["full"]), 4*len(cases))
        for i in range(0, len(t["local"]), 4):
            self.assertEqual(t["local"][i], t["local"][i+2])
            self.assertEqual(t["local"][i+1], t["local"][i+3])
        self.assertTrue(all("According to" not in text for text in t["old"]))

    def test_first_color_and_bag_strategies_cannot_pass_quartet(self):
        from breadcrumb_memory.binding_eval import score
        target = torch.tensor([[0, 1, 1, 0], [2, 3, 3, 2]])
        first_color = torch.tensor([[0, 0, 1, 1], [2, 2, 3, 3]])
        bag = torch.tensor([[0, 0, 0, 0], [2, 2, 2, 2]])
        self.assertEqual(score(first_color, target)["all_four"], 0)
        self.assertEqual(score(bag, target)["all_four"], 0)
        self.assertEqual(score(bag, target)["per_answer"], 0.5)
        self.assertEqual(score(target, target)["all_four"], 1)

    def test_gpt2_backbone_accepts_soft_memory_and_backpropagates(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        from breadcrumb_memory.lm_memory import student_logits
        model = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=32, n_layer=2, n_head=4,
                                          n_positions=64)).eval().requires_grad_(False)
        memory = torch.randn(2, 4, 32, requires_grad=True)
        local = {"input_ids": torch.randint(64, (2, 8)), "attention_mask": torch.ones(2, 8, dtype=torch.long)}
        student_logits(model, memory, local).sum().backward()
        self.assertGreater(memory.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
