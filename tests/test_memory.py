import unittest
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

from breadcrumb_memory.data import Episode, exact_retrieval, generate, retained_start
from breadcrumb_memory.experiment import Config, evaluate, run, wilson
from breadcrumb_memory.models import AssociativeRing, FullHistoryTeacher


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)

    def test_new_bindings_and_old_queries(self):
        rng = torch.Generator().manual_seed(123)
        e = generate(128, 40, 128, 32, 8, rng)
        self.assertTrue((e.support < 32).all())
        self.assertTrue((e.keys.sort(-1).values.diff(dim=-1) != 0).all())
        self.assertTrue(torch.equal(exact_retrieval(e), e.target))
        self.assertTrue((exact_retrieval(e, 32) == -1).all())
        other = generate(128, 40, 128, 32, 8, torch.Generator().manual_seed(456))
        self.assertFalse(torch.equal(e.values, other.values))

    def test_streaming_and_whole_history_have_same_state(self):
        model = AssociativeRing(32, 8, 4, 3, 2, 3)
        keys = torch.arange(19)[None, :]
        values = (keys * 3) % 8
        whole, rk, rv = model.encode(keys, values)
        streamed = model.empty_state(1, "cpu")
        model.write(streamed, keys[:, :5], values[:, :5])
        model.write(streamed, keys[:, 5:11], values[:, 5:11])
        model.write(streamed, keys[:, 11:16], values[:, 11:16])
        self.assertEqual(whole.count, streamed.count)
        for a, b in zip(whole.slots, streamed.slots):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(rk, keys[:, 16:])
        torch.testing.assert_close(rv, values[:, 16:])

    def test_overwrite_drops_old_records_including_partial_chunks(self):
        # Orthogonal keys let us inspect exact contents without neural ambiguity.
        model = AssociativeRing(32, 8, 32, 3, 2, 3)
        with torch.no_grad():
            model.embedding.weight.copy_(torch.eye(32))
        for length in range(4, 30):
            keys = torch.arange(length)[None, :]
            values = keys % 8
            state, _, _ = model.encode(keys, values)
            summed = torch.stack(state.slots).sum(0)[0]
            start = retained_start(length, 3, 2, 3)
            expected = torch.zeros_like(summed)
            for pos in range(start, max(0, length-3)):
                expected[pos, pos % 8] = 1
            torch.testing.assert_close(summed, expected)

    def test_memory_size_is_constant_as_history_grows(self):
        model = AssociativeRing(128, 8, 4, 3, 2, 3)
        for length in (4, 20, 100):
            keys = torch.arange(length)[None, :]
            state, _, _ = model.encode(keys, keys % 8)
            self.assertEqual(sum(x.numel()*x.element_size() for x in state.slots), model.state_bytes())

    def test_query_does_not_mutate_memory(self):
        model = AssociativeRing(32, 8, 4, 3, 2, 3)
        keys = torch.arange(19)[None, :]
        state, rk, rv = model.encode(keys, keys % 8)
        before = [x.detach().clone() for x in state.slots]
        model.read(state, rk, rv, torch.tensor([1]))
        model.read(state, rk, rv, torch.tensor([12]))
        for a, b in zip(before, state.slots):
            torch.testing.assert_close(a, b)

    def test_distillation_reaches_writer_but_not_frozen_teacher(self):
        teacher = FullHistoryTeacher(32, 8).requires_grad_(False)
        model = AssociativeRing(32, 8, 4, 3, 2, 3)
        e = generate(4, 8, 32, 8, 3, torch.Generator().manual_seed(3))
        with torch.no_grad():
            target = teacher(e.keys, e.values, e.query).softmax(-1)
        state, rk, rv = model.encode(e.keys, e.values)
        written_slots = [s for s in state.slots if s.requires_grad]
        for slot in written_slots:
            slot.retain_grad()
        loss = F.kl_div(model.read(state, rk, rv, e.query).log_softmax(-1), target, reduction="batchmean")
        loss.backward()
        self.assertTrue(any(s.grad is not None and s.grad.abs().sum() > 0 for s in written_slots))
        self.assertTrue(torch.isfinite(model.embedding.weight.grad).all())
        self.assertGreater(model.embedding.weight.grad[e.keys[:, :5]].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_shuffle_is_other_episode_memory(self):
        model = AssociativeRing(32, 8, 32, 3, 2, 3)
        with torch.no_grad():
            model.embedding.weight.copy_(torch.eye(32))
        keys = torch.arange(8).repeat(2, 1)
        values = torch.stack((torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)))
        state, rk, rv = model.encode(keys, values)
        q = torch.tensor([0, 0])
        self.assertEqual(model.read(state, rk, rv, q).argmax(-1).tolist(), [0, 1])
        self.assertEqual(model.read(state, rk, rv, q, "shuffled").argmax(-1).tolist(), [1, 0])

    def test_odd_evaluation_size_and_confidence_interval(self):
        cfg = Config(eval_examples=9, batch_size=4, eval_lengths=(16,))
        teacher = FullHistoryTeacher(cfg.n_keys, cfg.n_values)
        student = AssociativeRing(cfg.n_keys, cfg.n_values, cfg.key_dim, cfg.slots, cfg.chunk_size, cfg.fast_window)
        rows = evaluate(teacher, student, cfg, torch.device("cpu"))
        self.assertEqual([r["examples"] for r in rows], [9, 9])
        self.assertTrue(all(r["accuracy"]["exact_archive"] == 1 for r in rows))
        self.assertIsNone(wilson(0, 0))
        self.assertLess(wilson(9, 9)[0], 1)

    def test_time_limit_saves_partial_checkpoint_and_never_claims_completion(self):
        with tempfile.TemporaryDirectory() as path:
            cfg = Config(max_minutes=1e-9, teacher_steps=1, student_steps=1)
            with self.assertRaises(TimeoutError):
                run(cfg, path, "cpu")
            self.assertTrue((Path(path)/"checkpoint.pt").exists())
            self.assertTrue((Path(path)/"INTERRUPTED.txt").exists())
            self.assertFalse((Path(path)/"results.json").exists())
            saved = torch.load(Path(path)/"checkpoint.pt", weights_only=True, map_location="cpu")
            self.assertEqual(saved["status"], "interrupted")
            with self.assertRaises(FileExistsError):
                run(cfg, path, "cpu")


if __name__ == "__main__":
    unittest.main()
