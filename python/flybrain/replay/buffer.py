from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    left_eye: np.ndarray | None = None
    right_eye: np.ndarray | None = None
    next_left_eye: np.ndarray | None = None
    next_right_eye: np.ndarray | None = None


class ReplayBuffer:
    """Fixed-capacity transition buffer with uniform sampling."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self._storage: list[Transition] = []
        self._index = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, transition: Transition) -> None:
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._index] = transition
        self._index = (self._index + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self._storage):
            raise ValueError("batch_size exceeds buffer size")
        indices = np.random.choice(len(self._storage), size=batch_size, replace=False)
        return [self._storage[i] for i in indices]

    def sample_tensors(self, batch_size: int) -> dict[str, np.ndarray]:
        batch = self.sample(batch_size)
        return {
            "observation": np.stack([t.observation for t in batch]),
            "action": np.stack([t.action for t in batch]),
            "reward": np.asarray([t.reward for t in batch], dtype=np.float32),
            "next_observation": np.stack([t.next_observation for t in batch]),
            "done": np.asarray([t.done for t in batch], dtype=np.float32),
        }
