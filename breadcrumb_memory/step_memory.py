"""CPU-only causal simulation of circular memory and tool-step prefetch.

This is a scripted harness, not a trained model or a description of ChatGPT.
Run: python -m breadcrumb_memory.step_memory --output runs/step-memory
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import struct
from typing import Generic, TypeVar

T = TypeVar("T")
RECORD_BYTES = struct.calcsize("<qqqq")  # key, value, version, archive address
POINTER_BYTES = struct.calcsize("<qq")  # key, archive address
RING_METADATA_BYTES = 16  # next-write position and occupancy, both uint64
CONTROLLER_BYTES = 24  # current cue, step count, and spent call budget


class Ring(Generic[T]):
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.slots: list[T | None] = [None] * capacity
        self.next = 0
        self.count = 0
        self.writes = 0

    def append(self, item: T) -> T | None:
        old = self.slots[self.next]
        self.slots[self.next] = item
        self.next = (self.next + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        self.writes += 1
        return old

    def recent(self) -> list[T]:
        return [self.slots[(self.next - 1 - i) % self.capacity]
                for i in range(self.count)]  # type: ignore[misc]


@dataclass(frozen=True)
class Record:
    key: int
    value: int
    version: int
    address: int


@dataclass(frozen=True)
class Event:
    kind: str
    key: int | None = None
    value: int | None = None


@dataclass(frozen=True)
class Config:
    seed: int = 17
    episodes_per_scenario: int = 100
    fast_slots: int = 4
    slow_slots: int = 16
    read_budget: int = 1
    prelude_records: int = 64


POLICIES = (
    "dual_scheduled", "dual_periodic", "dual_on_demand", "dual_no_refresh",
    "single_scheduled", "single_on_demand", "single_no_refresh",
    "exact_on_demand", "dual_wrong_pointer",
)
SCENARIOS = ("matched_timing", "cue", "no_cue", "changed_task", "stale_update", "expired_pointer")


class MemorySystem:
    """Policies only receive past/current events; evaluation truth stays outside."""
    def __init__(self, policy: str, config: Config):
        if policy not in POLICIES:
            raise ValueError(policy)
        self.policy, self.config = policy, config
        # Both layouts reserve the same budget, including ring cursor metadata.
        self.budget_bytes = (config.fast_slots * RECORD_BYTES
                             + config.slow_slots * POINTER_BYTES
                             + 2 * RING_METADATA_BYTES)
        single = policy.startswith("single")
        slots = ((self.budget_bytes - RING_METADATA_BYTES) // RECORD_BYTES
                 if single else config.fast_slots)
        self.fast: Ring[Record] = Ring(slots)
        self.slow: Ring[tuple[int, int]] | None = None if single else Ring(config.slow_slots)
        self.allocated_bytes = slots * RECORD_BYTES + RING_METADATA_BYTES
        if self.slow is not None:
            self.allocated_bytes += config.slow_slots * POINTER_BYTES + RING_METADATA_BYTES
        self.archive: list[Record] = []
        # Index is available ONLY to exact baseline; others must use resident cues.
        self.index: dict[int, int] = {}
        self.cue: int | None = None
        self.steps = self.calls = self.reads = self.answer_calls = 0
        self.matching_key_reads = 0
        self.source_ages: list[int] = []

    def _insert(self, record: Record) -> None:
        evicted = self.fast.append(record)
        if evicted is not None and self.slow is not None:
            self.slow.append((evicted.key, evicted.address))

    def _resident(self, key: int) -> Record | None:
        matches = [r for r in self.fast.recent() if r.key == key]
        return max(matches, key=lambda r: r.version, default=None)

    def _address(self, key: int) -> int | None:
        if self.policy == "exact_on_demand":
            return self.index.get(key)
        addresses = [r.address for r in self.fast.recent() if r.key == key]
        if self.slow is not None:
            addresses += [address for k, address in self.slow.recent() if k == key]
        return max(addresses, default=None)

    def refresh(self, key: int | None, *, at_answer: bool = False) -> None:
        if key is None or self.calls >= self.config.read_budget:
            return
        self.calls += 1
        self.answer_calls += int(at_answer)
        address = self._address(key)
        if address is None:
            return
        if self.policy == "dual_wrong_pointer":
            address = (address + 1) % len(self.archive)
        self.reads += 1
        self.matching_key_reads += int(self.archive[address].key == key)
        self.source_ages.append(len(self.archive) - 1 - address)
        self._insert(self.archive[address])

    def process(self, event: Event) -> Record | None:
        if event.kind == "write":
            assert event.key is not None and event.value is not None
            # Version comes from the observable environment, not a model lookup.
            previous = self.index.get(event.key)
            version = self.archive[previous].version + 1 if previous is not None else 1
            record = Record(event.key, event.value, version, len(self.archive))
            self.archive.append(record)
            self.index[event.key] = record.address
            self._insert(record)
        elif event.kind == "cue":
            self.cue = event.key
        elif event.kind == "step":
            self.steps += 1
            if self.policy.endswith("scheduled"):
                self.refresh(self.cue)
            elif self.policy == "dual_periodic" and self.steps % 3 == 0:
                self.refresh(self.cue)
        elif event.kind == "query":
            assert event.key is not None
            if self.policy.endswith("on_demand") or self.policy == "dual_wrong_pointer":
                self.refresh(event.key, at_answer=True)
            return self._resident(event.key)
        else:
            raise ValueError(event.kind)
        return None

    def metrics(self) -> dict:
        return {
            "persistent_budget_bytes": self.budget_bytes + CONTROLLER_BYTES,
            "ring_allocated_bytes": self.allocated_bytes,
            "controller_bytes": CONTROLLER_BYTES,
            "unused_budget_bytes": self.budget_bytes - self.allocated_bytes,
            "archive_bytes": len(self.archive) * RECORD_BYTES,
            "external_index_bytes": (len(self.index) * POINTER_BYTES
                                     if self.policy == "exact_on_demand" else 0),
            "environment_version_index_bytes": len(self.index) * POINTER_BYTES,
            "archive_records": len(self.archive),
            "refresh_calls": self.calls,
            "records_reread": self.reads,
            "matching_key_reads": self.matching_key_reads,
            "bytes_reread": self.reads * RECORD_BYTES,
            "index_probe_payload_bytes": (self.calls * POINTER_BYTES
                                          if self.policy == "exact_on_demand" else 0),
            "answer_time_calls": self.answer_calls,
            "fast_ring_wraps": self.fast.writes // self.fast.capacity,
            "slow_ring_wraps": (self.slow.writes // self.slow.capacity if self.slow else 0),
            "mean_retrieved_source_age": (sum(self.source_ages) / len(self.source_ages)
                                          if self.source_ages else 0),
        }


def make_episode(config: Config, scenario: str, rng: random.Random) -> list[Event]:
    """Environment generates complete episodes; policies consume them one at a time."""
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    n = config.prelude_records
    if n < 48:
        raise ValueError("prelude_records must be at least 48")
    # Random, independent values prevent encoding key-to-value rules.
    # Opaque identities must not reveal archive positions (key == address would
    # let an untested direct-address baseline bypass breadcrumbs entirely).
    keys = rng.sample(range(1_000_000, 9_000_000), n + 24)
    events = [Event("write", keys[i], rng.randrange(1_000_000)) for i in range(n)]
    distance = rng.choice((6, 10, 14, 18))
    target = keys[n - distance]
    if scenario == "expired_pointer":
        target = keys[n - 40]
    if scenario != "no_cue":
        events.append(Event("cue", target))
    events.append(Event("step"))
    if scenario == "changed_task":
        target = keys[n - rng.choice((7, 11, 15, 19))]
        events.append(Event("cue", target))
    if scenario == "stale_update":
        events.append(Event("write", target, rng.randrange(1_000_000)))
    # Vary delay independently; prefetch can be overwritten before consumption.
    delay = rng.choice((0, 2, 8, 24))
    if scenario == "matched_timing":
        delay = rng.choice((0, 2))
    for i in range(delay):
        events.append(Event("write", keys[n + i], rng.randrange(1_000_000)))
        events.append(Event("step"))
    events.append(Event("query", target))
    return events


def evaluate_episode(events: list[Event], policy: str, config: Config) -> dict:
    system = MemorySystem(policy, config)
    truth: dict[int, tuple[int, int]] = {}
    correct = stale = answers = 0
    for event in events:
        if event.kind == "write":
            prev = truth.get(event.key, (0, 0))
            truth[event.key] = (event.value, prev[1] + 1)
        answer = system.process(event)
        if event.kind == "query":
            expected_value, expected_version = truth[event.key]
            answers += int(answer is not None)
            correct += int(answer is not None and answer.value == expected_value
                           and answer.version == expected_version)
            stale += int(answer is not None and answer.version < expected_version)
    return {**system.metrics(), "correct": correct, "stale": stale,
            "answered": answers, "queries": sum(e.kind == "query" for e in events)}


def run(config: Config) -> dict:
    rng = random.Random(config.seed)
    groups = []
    for scenario in SCENARIOS:
        episodes = [make_episode(config, scenario, rng) for _ in range(config.episodes_per_scenario)]
        for policy in POLICIES:
            rows = [evaluate_episode(e, policy, config) for e in episodes]
            means = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
            groups.append({"scenario": scenario, "policy": policy, **means,
                           "accuracy": means["correct"] / means["queries"],
                           "stale_rate": means["stale"] / means["queries"]})
    return {"experiment": "scripted_causal_step_memory", "config": asdict(config),
            "limitations": ["No learned model, language, real tool execution, or actual async overlap.",
                            "Exact keys are explicit cues; task information can change or be absent.",
                            "Equal maximum call/read budgets; actual utilization can differ.",
                            "Matched-timing stratum isolates equal actual reads for dual scheduled vs demand.",
                            "Packed logical byte accounting excludes Python/interpreter overhead.",
                            "Archive grows; resident ring budget stays constant.",
                            "Answer-time calls are a latency proxy, not measured end-to-end latency."],
            "results": groups}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be a new or empty directory")
    result = run(Config(seed=args.seed, episodes_per_scenario=args.episodes))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Scripted circular-buffer experiment", "", "CPU simulation; no trained LLM.", "",
             "| Scenario | Policy | Correct | Stale | Reads | Answer-time calls |", "|---|---|---:|---:|---:|---:|"]
    for row in result["results"]:
        lines.append(f"| {row['scenario']} | {row['policy']} | {row['accuracy']:.1%} | "
                     f"{row['stale_rate']:.1%} | {row['records_reread']:.2f} | {row['answer_time_calls']:.2f} |")
    lines += ["", "Limitations:", ""] + ["- " + item for item in result["limitations"]]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "episodes_per_scenario": args.episodes,
                      "policies": len(POLICIES)}, indent=2))


if __name__ == "__main__":
    main()
