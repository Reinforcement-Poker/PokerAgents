# %%
import jax
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Array, Float32, Scalar


ActionsObservation = Float32[Array, "24 n_players+2 n_actions"]
CardsObservation = Float32[Array, "4 13 6"]
Policy = Float32[Array, "n_actions"]
Value = Float32[Scalar, ""]
HandRank = Float32[Scalar, ""]


class AlphaHoldem(nn.Module):
    @nn.compact
    def __call__(
        self,
        actions_observation: ActionsObservation,
        cards_observation: CardsObservation,
        train: bool = True,
    ) -> tuple[Policy, Value, HandRank]:
        policy = PolicyNet()(actions_observation, cards_observation, train)
        value, hand_rank = ValueNet()(actions_observation, cards_observation, train)

        return policy, value, hand_rank


class PolicyNet(nn.Module):
    @nn.compact
    def __call__(
        self,
        actions_observation: ActionsObservation,
        cards_observation: CardsObservation,
        train: bool = True,
    ) -> Policy:
        actions_encoding = ActionsEncoder()(actions_observation, train)
        cards_encoding = CardsEncoder()(cards_observation, train)
        x = jnp.concatenate([actions_encoding, cards_encoding])

        n_actions = actions_observation.shape[1]
        for i in range(1, 50, 10):
            x = nn.Dense(
                features=(n_actions + 3) * 50 // i,
                kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
                bias_init=nn.initializers.constant(0.0),
            )(x)
            # x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.tanh(x)

        x = nn.Dense(
            n_actions + 3,
            kernel_init=nn.initializers.orthogonal(scale=0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        policy = nn.softmax(x)

        return policy


class ValueNet(nn.Module):
    @nn.compact
    def __call__(
        self,
        actions_observation: ActionsObservation,
        cards_observation: CardsObservation,
        train: bool = True,
    ) -> tuple[Value, HandRank]:
        actions_encoding = ActionsEncoder()(actions_observation, train)
        # jax.debug.print("actions_encoding: {}", actions_encoding.shape)
        cards_encoding = CardsEncoder()(cards_observation, train)
        # jax.debug.print("cards_encoding: {}", cards_encoding.shape)
        x = jnp.concatenate([actions_encoding, cards_encoding])
        # jax.debug.print("x: {}", x.shape)

        for i in range(6):
            # jax.debug.print("dense i: {}", i)
            x = nn.Dense(features=128 // (2**i))(x)
            # x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.leaky_relu(x)

        value, hand_rank = nn.Dense(2)(x)

        return value, hand_rank


class ActionsEncoder(nn.Module):
    @nn.compact
    def __call__(
        self, actions_observation: ActionsObservation, train: bool = True
    ) -> Float32[Array, "64"]:
        x = actions_observation
        x = nn.Conv(features=24, kernel_size=(3, 3), padding="SAME")(x)
        # x = nn.BatchNorm(use_running_average=not train)(x)
        x: Array = nn.tanh(x)

        for i in range(5, 7):
            x = nn.Conv(features=2**i, kernel_size=(3, 1), padding="VALID")(x)
            x: Array = nn.tanh(x)
            x = nn.Conv(features=2**i, kernel_size=(3, 3), padding="SAME")(x)
            # x = nn.BatchNorm(use_running_average=not train)(x)
            x: Array = nn.tanh(x)

        x = jnp.mean(x, axis=(0, 1))
        return x


class CardsEncoder(nn.Module):
    @nn.compact
    def __call__(
        self, cards_observation: CardsObservation, train: bool = True
    ) -> Float32[Array, "64"]:
        x = cards_observation

        x = nn.Conv(features=24, kernel_size=(3, 3), padding="SAME")(x)
        # x = nn.BatchNorm(use_running_average=not train)(x)
        x: Array = nn.tanh(x)

        for i in range(5, 7):
            x = nn.Conv(features=2**i, kernel_size=(1, 3), padding="VALID")(x)
            x: Array = nn.tanh(x)
            x = nn.Conv(features=2**i, kernel_size=(1, 3), padding="VALID")(x)
            x: Array = nn.tanh(x)
            x = nn.Conv(features=2**i, kernel_size=(3, 3), padding="SAME")(x)
            # x = nn.BatchNorm(use_running_average=not train)(x)
            x: Array = nn.tanh(x)

        x = jnp.mean(x, axis=(0, 1))
        return x


if __name__ == "__main__":
    m = AlphaHoldem()
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)
    actions = jax.random.uniform(key1, (24, 5, 8))
    cards = jax.random.uniform(key2, (4, 13, 6))
    params = m.init(key3, actions, cards)
    (policy, value), new_batch_stats = m.apply(params, actions, cards, mutable=["batch_stats"])
    print(m.tabulate(key3, actions, cards, console_kwargs=dict(width=200)))
# %%
