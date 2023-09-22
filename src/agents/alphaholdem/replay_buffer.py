import jax
import jax.numpy as jnp
import math
from flax import struct
from jaxtyping import Array, Scalar, Float, Int
from src.agents.alphaholdem.model import ActionsObservation, CardsObservation


# Observations and reward of a single agent along a hand
@struct.dataclass
class BufferRecord:
    actions_observations: Float[Array, "hand_len 24 n_actions n_players+2"]
    cards_observations: Float[Array, "hand_len 4 13 6"]
    actions_taken: Int[Array, "hand_len"]
    reward: Float[Scalar, ""]
    hand_scores: Float[Array, "hand_len"]


class ReplayBuffer:
    def __init__(self, key: jax.random.KeyArray):
        self.iter_key = jax.random.split(key, 2)
        self.buffer: list[BufferRecord] = []

    def add_agent_play(
        self,
        record: BufferRecord,
    ):
        self.buffer.append(record)

    def reset(self):
        self.buffer = []

    def batch_iter(self, batch_size: int):
        # indices = jax.random.permutation(self.iter_key, jnp.arange(len(self.buffer))).astype(int)
        indices = jnp.arange(len(self.buffer)).astype(int)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            yield [self.buffer[idx] for idx in batch_indices]

    def n_batches(self, batch_size: int) -> int:
        return math.ceil(len(self.buffer) / batch_size)
