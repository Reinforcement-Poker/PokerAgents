import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float
from src.agents.alphaholdem.model import ActionsObservation, CardsObservation


@struct.dataclass
class BufferRecord:
    actions_observations: Float[Array, "hand_len 24 n_actions n_players+2"]
    cards_observations: Float[Array, "hand_len 4 13 6"]
    rewads: Float[Array, "n_players"]


class ReplayBuffer:
    def __init__(self, size: int, key: jax.random.KeyArray):
        self.size = size
        self.update_key: jax.random.KeyArray
        self.iter_key: jax.random.KeyArray
        self.update_key, self.iter_key = jax.random.split(key, 2)
        self.buffer: list[BufferRecord] = []

    def update(
        self,
        actions_observations: list[ActionsObservation],
        cards_observations: list[CardsObservation],
        rewards: list[float],
    ):
        if len(self.buffer) >= self.size:
            i = jax.random.randint(self.update_key, (1,), 0, len(self.buffer))
            _, self.update_key = jax.random.split(self.update_key)
            self.buffer.pop(i)

        self.buffer.append(
            BufferRecord(
                actions_observations=jnp.array(actions_observations),
                cards_observations=jnp.array(cards_observations),
                rewads=jnp.array(rewards),
            )
        )

    def batch_iter(self, batch_size: int):
        indices = jax.random.permutation(self.iter_key, jnp.arange(len(self.buffer))).astype(int)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            yield [self.buffer[idx] for idx in batch_indices]
