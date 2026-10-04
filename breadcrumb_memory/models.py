"""Content-attention teacher and learned linear associative ring.

These are tiny task-specific networks, not pretrained language models or TTCD.
The value encoding is deliberately fixed one-hot; the key feature map is learned.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class FullHistoryTeacher(nn.Module):
    def __init__(self, n_keys, n_values, dim=64):
        super().__init__()
        self.embedding = nn.Embedding(n_keys, dim)
        self.log_scale = nn.Parameter(torch.tensor(math.log(8.0)))
        self.n_values = n_values

    def forward(self, keys, values, query):
        k = F.normalize(self.embedding(keys), dim=-1)
        q = F.normalize(self.embedding(query), dim=-1)
        score = torch.einsum("btd,bd->bt", k, q) * self.log_scale.exp().clamp(max=100)
        attention = score.softmax(dim=-1)
        probs = attention.new_zeros(keys.shape[0], self.n_values)
        probs = probs.scatter_add(1, values, attention)
        return probs.clamp_min(1e-9).log()


@dataclass
class RingState:
    slots: list
    count: int = 0


class AssociativeRing(nn.Module):
    def __init__(self, n_keys, n_values, key_dim, slots, chunk_size, fast_window):
        super().__init__()
        self.embedding = nn.Embedding(n_keys, key_dim)
        self.log_scale = nn.Parameter(torch.tensor(math.log(4.0)))
        self.n_values = n_values
        self.key_dim = key_dim
        self.slots = slots
        self.chunk_size = chunk_size
        self.fast_window = fast_window

    def empty_state(self, batch_size, device):
        return RingState([torch.zeros(batch_size, self.key_dim, self.n_values,
                                     device=device, dtype=self.embedding.weight.dtype)
                          for _ in range(self.slots)])

    def feature(self, keys):
        return F.normalize(self.embedding(keys), dim=-1)

    def write(self, state, keys, values):
        """Write chronological records. This API deliberately has no query argument.

        A slot accumulates chunk_size outer products, then the next slot is used.
        Wrapping clears the oldest slot. Functional assignments preserve autograd.
        No raw historical keys or values are retained in RingState.
        """
        for i in range(keys.shape[1]):
            slot = (state.count // self.chunk_size) % self.slots
            value = F.one_hot(values[:, i], self.n_values).to(self.embedding.weight.dtype)
            update = self.feature(keys[:, i]).unsqueeze(-1) * value.unsqueeze(1)
            state.slots[slot] = (update if state.count % self.chunk_size == 0
                                 else state.slots[slot] + update)
            state.count += 1
        return state

    def encode(self, keys, values):
        old_count = max(0, keys.shape[1] - self.fast_window)
        state = self.empty_state(keys.shape[0], keys.device)
        state = self.write(state, keys[:, :old_count], values[:, :old_count])
        return state, keys[:, old_count:], values[:, old_count:]

    def read(self, state, recent_keys, recent_values, query, mode="ring"):
        memory = torch.stack(state.slots, dim=1).sum(dim=1)
        if mode == "no_memory":
            memory = torch.zeros_like(memory)
        elif mode == "shuffled":
            # Other episodes' memory; keep this episode's recent context and query.
            memory = memory.roll(1, dims=0)
        elif mode != "ring":
            raise ValueError(f"unknown memory mode: {mode}")
        recent = F.one_hot(recent_values, self.n_values).to(memory.dtype)
        memory = memory + torch.einsum("btd,btv->bdv", self.feature(recent_keys), recent)
        score = torch.einsum("bd,bdv->bv", self.feature(query), memory)
        return score * self.log_scale.exp().clamp(max=100)

    def forward(self, keys, values, query, mode="ring"):
        state, recent_keys, recent_values = self.encode(keys, values)
        return self.read(state, recent_keys, recent_values, query, mode)

    def state_bytes(self, batch_size=1):
        # Persistent slow state only. Excludes recent IDs, model, temporaries, autograd.
        return batch_size * self.slots * self.key_dim * self.n_values * self.embedding.weight.element_size()
