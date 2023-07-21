# %%
import jax
import flax.linen as nn
from jaxtyping import Array, Float32, Scalar


ActionsObservation = Float32[Array, "24 n_actions n_players+2"]
CardsObservation = Float32[Array, "4 13 6"]
Policy = Float32[Array, "n_actions"]
Value = Float32[Scalar, ""]


class AlphaHoldem(nn.Module):
    @nn.compact
    def __call__(
        self,
        actions_observation: ActionsObservation,
        cards_observation: CardsObservation,
        train: bool = True,
    ) -> tuple[Policy, Value]:
        actions_encoding = ActionsEncoder()(actions_observation, train)
        cards_encoding = CardsEncoder()(cards_observation, train)
        x = jax.numpy.concatenate([actions_encoding, cards_encoding])

        n_actions = actions_observation.shape[1]
        # Basic actions + raise{1/2, 2/3, 1, 3/2} + value
        out_dim = n_actions + 3 + 1
        for i in range(1, 50, 5):
            x = nn.Dense(features=(out_dim) * 50 // i)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.leaky_relu(x)

        x = nn.Dense(out_dim)(x)
        policy = nn.softmax(x[:-1])
        value = nn.sigmoid(x[-1]) * 200 - 100

        return policy, value


class ActionsEncoder(nn.Module):
    @nn.compact
    def __call__(
        self, actions_observation: ActionsObservation, train: bool = True
    ) -> Float32[Array, "64"]:
        x = actions_observation
        x = nn.Conv(features=24, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x: Array = nn.leaky_relu(x)

        for i in range(5, 7):
            x = nn.Conv(features=2**i, kernel_size=(3, 1), padding="VALID")(x)
            x: Array = nn.leaky_relu(x)
            x = nn.Conv(features=2**i, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x: Array = nn.leaky_relu(x)

        x = jax.numpy.mean(x, axis=(0, 1))
        return x


class CardsEncoder(nn.Module):
    @nn.compact
    def __call__(
        self, cards_observation: CardsObservation, train: bool = True
    ) -> Float32[Array, "64"]:
        x = cards_observation

        x = nn.Conv(features=24, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x: Array = nn.leaky_relu(x)

        for i in range(5, 7):
            x = nn.Conv(features=2**i, kernel_size=(1, 3), padding="VALID")(x)
            x: Array = nn.leaky_relu(x)
            x = nn.Conv(features=2**i, kernel_size=(1, 3), padding="VALID")(x)
            x: Array = nn.leaky_relu(x)
            x = nn.Conv(features=2**i, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x: Array = nn.leaky_relu(x)

        x = jax.numpy.mean(x, axis=(0, 1))
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
