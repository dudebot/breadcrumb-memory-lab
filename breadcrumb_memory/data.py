from dataclasses import dataclass

import torch


@dataclass
class Episode:
    keys: torch.Tensor
    values: torch.Tensor
    query: torch.Tensor
    target: torch.Tensor
    support: torch.Tensor

    def to(self, device):
        return Episode(*(x.to(device) for x in self.__dict__.values()))


def generate(batch_size, length, n_keys, n_values, fast_window, generator, query_mode="old"):
    """Unique keys and independently sampled values; no future query enters a writer.

    The vocabulary is shared across splits, but every episode has a newly randomized
    key/value assignment. This tests new bindings, not unseen vocabulary.
    """
    if not 0 < length <= n_keys:
        raise ValueError("length must be in [1, n_keys] (unique-key task)")
    if query_mode not in {"old", "recent", "any"}:
        raise ValueError("query_mode must be old, recent, or any")
    keys = torch.rand(batch_size, n_keys, generator=generator).argsort(dim=-1)[:, :length]
    values = torch.randint(n_values, (batch_size, length), generator=generator)
    if query_mode == "old":
        lo, hi = 0, max(1, length - fast_window)
    elif query_mode == "recent":
        lo, hi = max(0, length - fast_window), length
    else:
        lo, hi = 0, length
    support = torch.randint(lo, hi, (batch_size,), generator=generator)
    rows = torch.arange(batch_size)
    return Episode(keys, values, keys[rows, support], values[rows, support], support)


def retained_start(length, fast_window, chunk_size, slots):
    evicted = max(0, length - fast_window)
    chunks = (evicted + chunk_size - 1) // chunk_size
    return max(0, chunks - slots) * chunk_size


def exact_retrieval(episode, start=0):
    """Transparent keyed-archive control; -1 means evidence was not retained."""
    keys, values = episode.keys[:, start:], episode.values[:, start:]
    matches = keys == episode.query[:, None]
    positions = matches.long().argmax(dim=-1)
    predictions = values.gather(1, positions[:, None]).squeeze(1)
    return predictions.masked_fill(~matches.any(dim=-1), -1)
