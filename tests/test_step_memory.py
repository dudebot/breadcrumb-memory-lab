import unittest

from breadcrumb_memory.step_memory import (
    Config, Event, MemorySystem, POLICIES, Ring, evaluate_episode, make_episode, run,
)
import random


class StepMemoryTests(unittest.TestCase):
    def test_ring_really_overwrites_and_orders_newest_first(self):
        ring = Ring(3)
        self.assertIsNone(ring.append(1))
        ring.append(2)
        ring.append(3)
        self.assertEqual(ring.append(4), 1)
        self.assertEqual(ring.recent(), [4, 3, 2])
        self.assertEqual(len(ring.slots), 3)

    def test_refresh_boundary_is_causal_and_prefix_invariant(self):
        prefix = [Event("write", i, i * 9) for i in range(20)]
        prefix += [Event("cue", 10)]
        first = MemorySystem("dual_scheduled", Config())
        second = MemorySystem("dual_scheduled", Config())
        for event in prefix:
            first.process(event)
            second.process(event)
        self.assertEqual(first.calls, 0)
        first.process(Event("step"))
        second.process(Event("step"))
        self.assertEqual(first.fast.recent(), second.fast.recent())
        self.assertEqual(first.calls, 1)
        self.assertEqual(first._resident(10).value, 90)
        # Different future requests cannot affect the identical pre-query state.
        first.process(Event("query", 10))
        second.process(Event("query", 11))
        self.assertEqual(first.fast.recent(), second.fast.recent())

    def test_latest_version_survives_stale_pointer_duplicates(self):
        system = MemorySystem("dual_on_demand", Config())
        for event in [Event("write", 1, 10)] + [Event("write", i, 0) for i in range(2, 9)]:
            system.process(event)
        system.process(Event("write", 1, 77))
        for i in range(9, 14):
            system.process(Event("write", i, 0))
        result = system.process(Event("query", 1))
        self.assertEqual((result.value, result.version), (77, 2))

    def test_equal_budget_and_hard_read_cap(self):
        systems = [MemorySystem(policy, Config()) for policy in POLICIES]
        self.assertEqual(len({s.metrics()["persistent_budget_bytes"] for s in systems}), 1)
        for system in systems:
            for i in range(30):
                system.process(Event("write", i, i))
            for _ in range(5):
                system.refresh(20)
            self.assertEqual(system.calls, 1)
            self.assertLessEqual(system.reads, 1)
            self.assertLessEqual(system.allocated_bytes, system.budget_bytes)

    def test_no_cue_does_not_access_future_query(self):
        system = MemorySystem("dual_scheduled", Config())
        for i in range(30):
            system.process(Event("write", i, i))
            system.process(Event("step"))
        self.assertEqual(system.calls, 0)
        self.assertIsNone(system.process(Event("query", 20)))
        self.assertEqual(system.calls, 0)

    def test_controls_and_exact_lookup(self):
        events = [Event("write", i, i + 100) for i in range(64)]
        events += [Event("query", 50)]
        self.assertEqual(evaluate_episode(events, "dual_on_demand", Config())["correct"], 1)
        self.assertEqual(evaluate_episode(events, "dual_wrong_pointer", Config())["correct"], 0)
        self.assertEqual(evaluate_episode(events, "dual_no_refresh", Config())["correct"], 0)
        self.assertEqual(evaluate_episode(events, "single_on_demand", Config())["correct"], 0)
        events[-1] = Event("query", 0)
        result = evaluate_episode(events, "exact_on_demand", Config())
        self.assertEqual(result["correct"], 1)
        self.assertGreater(result["external_index_bytes"], 0)

    def test_repeated_turnover_bounded_and_archive_costed(self):
        system = MemorySystem("dual_scheduled", Config())
        for i in range(1000):
            system.process(Event("write", i, i))
        stats = system.metrics()
        self.assertEqual(len(system.fast.slots), 4)
        self.assertEqual(len(system.slow.slots), 16)
        self.assertEqual(stats["archive_bytes"], 32000)
        self.assertGreater(stats["slow_ring_wraps"], 10)
        self.assertIsNone(system._address(0))

    def test_run_reproducible(self):
        config = Config(episodes_per_scenario=2)
        self.assertEqual(run(config), run(config))

    def test_generated_keys_do_not_encode_archive_addresses(self):
        events = make_episode(Config(), "cue", random.Random(17))
        keys = [e.key for e in events if e.kind == "write"]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(key >= 1_000_000 for key in keys))
        self.assertNotEqual(keys, sorted(keys))

    def test_matched_timing_has_equal_actual_reads_and_different_critical_path(self):
        rng = random.Random(3)
        for _ in range(20):
            events = make_episode(Config(), "matched_timing", rng)
            scheduled = evaluate_episode(events, "dual_scheduled", Config())
            demand = evaluate_episode(events, "dual_on_demand", Config())
            self.assertEqual(scheduled["records_reread"], 1)
            self.assertEqual(demand["records_reread"], 1)
            self.assertEqual(scheduled["correct"], 1)
            self.assertEqual(demand["correct"], 1)
            self.assertEqual(scheduled["answer_time_calls"], 0)
            self.assertEqual(demand["answer_time_calls"], 1)


if __name__ == "__main__":
    unittest.main()
