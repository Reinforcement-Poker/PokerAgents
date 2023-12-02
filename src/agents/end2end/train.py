import jax
import jax.numpy as jnp
import numpy as np
from clu import metrics
import flax
from flax import linen as nn
import optax
import pgx
from clearml import Task, TaskTypes
from typing_extensions import Self


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


class PolicyModel(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        x = observation.astype(jnp.float32)
        x = nn.Dense(features=8)(x)
        x = nn.relu(x)
        x = nn.Dense(features=16)(x)
        x = nn.relu(x)
        x = nn.Dense(features=16)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=16)(x)
        x = nn.relu(x)
        x = nn.Dense(features=16)(x)
        x = nn.relu(x)
        x = nn.Dense(features=8)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        x = nn.softmax(x)
        return x


@flax.struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


@flax.struct.dataclass
class TrainState:
    module: nn.Module = flax.struct.field(pytree_node=False)
    models_params: list[nn.FrozenDict]
    optimizer: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    opt_states: list[optax.OptState]
    metrics: Metrics
    policy_matrix: jax.Array

    @classmethod
    def create(
        cls,
        module: nn.Module,
        models_params: list[nn.FrozenDict],
        optimizer: optax.GradientTransformation,
    ) -> Self:
        return cls(
            module=module,
            models_params=models_params,
            optimizer=optimizer,
            opt_states=[optimizer.init(p) for p in models_params],
            metrics=Metrics.empty(),
        )


def train():
    hyper_params = {
        "seed": 12345,
        "iterations": 10000,
        "hands_per_iter": 128,
        "game_samples": 20,
        "batch_size": 20,
        "lr": 1e-3,
        "grad_clip": 10.0,
    }

    task = Task.init(
        project_name="ReinforcementPoker/End2end",
        task_type=TaskTypes.training,
        task_name="Test end2end",
        tags=["kuhn", "test"],
    )
    task.connect(hyper_params)
    # task.execute_remotely(queue_name="vm-bruno")

    model = PolicyModel(n_actions=4)
    train_state = TrainState.create(
        module=model,
        models_params=[
            model.init(jax.random.PRNGKey(hyper_params["seed"]), jnp.ones(7))["params"]
            for _ in range(2)
        ],
        optimizer=optax.sgd(hyper_params["lr"]),
    )

    train_key = jax.random.PRNGKey(hyper_params["seed"])
    for i in range(hyper_params["iterations"]):
        train_state.replace(metrics=train_state.metrics.empty())
        train_state: TrainState = train_step(train_state, train_key)
        _, train_key = jax.random.split(train_key)

        print(f"loss={train_state.metrics.loss}")
        task.get_logger().report_scalar(
            title="train | loss",
            series="loss",
            value=float(train_state.metrics.loss.total),
            iteration=i,
        )
        task.get_logger().report_confusion_matrix(
            title="Policy matrix",
            series="train",
            iteration=i,
            matrix=train_state.policy_matrix,
            xaxis="Action",
            yaxis="Initial card",
            xlabels=["Fold", "Check", "Call", "Bet"],  # TODO: Ordenar bien
            ylabels=["Jack", "Queen", "King"],
        )


def train_step(state: TrainState, key: jax.Array) -> TrainState:
    def loss_fn(params):
        rewards, state = self_play(
            train_state=state,
            params=params,
            n_hands=10_000,
            n_samples=10,
            key=key,
        )

        return -jnp.mean(rewards), state

    # Primero hacer que solo aprenda un jugador
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_value, state), grads = grad_fn(state.models_params[0])
    updates, opt_state = state.optimizer.update(grads, state.opt_states[0])
    norm = optax.global_norm(updates)
    print("Norm:", norm)
    params = optax.apply_updates(state.models_params[0], updates)  # type: ignore
    metrics = state.metrics.single_from_model_output(loss=loss_value)
    state = state.replace(
        opt_states=[opt_state, state.opt_states[1]],
        models_params=[params, state.models_params[1]],
        metrics=metrics,
    )
    return state


# TODO: Pensar estrategia para samplear sin explosión de estados
# Hacer policy * resultado de haber realizado cada acción
# Habrá que pensar como podar ramas ¿considerar solo n=2 acciones aleatorias cada iteración?
# Además solo hay que tener en cuenta las acciones legales
# Si se superan las 10 iteraciones (<=1024 estados) se samplea una sola acción siguiendo la policy
# TODO: Usar jax.lax.while_loop
def self_play(
    train_state: TrainState, params, n_hands: int, n_samples: int, key: jax.Array
):
    env = pgx.make("kuhn_poker")
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))

    _, subkey1, subkey2 = jax.random.split(key, 3)

    keys_init = jax.random.split(subkey1, n_hands * 2)
    state: pgx.State = init(keys_init)
    terminated = np.array(state.terminated)
    truncated = np.array(state.truncated)
    decision_weights = None
    i = 0
    while not (terminated | truncated).all():
        print(i)
        policy: jax.Array = jax.vmap(
            lambda player, obs: train_state.module.apply(
                {
                    "params": jax.lax.cond(
                        player == 0,
                        lambda: params,
                        lambda: train_state.models_params[1],
                    )
                },
                obs,
            )
        )(state.current_player, state.observation)
        policy = policy * state.legal_action_mask
        policy = policy / jnp.sum(policy, axis=1)[:, jnp.newaxis]
        if i == 0:
            jax.debug.print("Policy: {}", policy.shape)
            sample_prob = policy[::2].at[policy[::2] != 0].set(1.0)
            sample_prob /= sample_prob.sum()
            jax.debug.print("Sample prob: {}", sample_prob.shape)
            action = jax.vmap(
                lambda p: jax.random.choice(
                    subkey2,
                    jnp.arange(env.num_actions),
                    p=p,
                    shape=(2,),
                    replace=False,
                )
            )(sample_prob)
            decision_weights = policy[::2][
                jnp.arange(policy.shape[0] // 2)[:, None], action
            ].reshape((-1,))
            train_state = train_state.replace(
                policy_matrix=jnp.array(
                    [
                        policy[jack].mean(axis=0),
                        policy[queen].mean(axis=0),
                        policy[king].mean(axis=0),
                    ]
                )
            )
        else:
            action = jax.vmap(
                lambda p: jax.random.choice(
                    subkey2,
                    jnp.arange(env.num_actions),
                    p=p,
                    shape=(1,),
                )
            )(policy)
        action = jnp.reshape(action, (-1,))
        # action = jnp.repeat(action, (n_hands * (n_samples**4)) // len(action), axis=0)
        state = step(state, action)
        terminated = np.array(state.terminated)
        truncated = np.array(state.truncated)
        i += 1

    return state.rewards[:, 0] * decision_weights, train_state


if __name__ == "__main__":
    train()
    # model = PolicyModel(n_actions=4)
    # train_state = TrainState.create(
    #     module=model,
    #     models_params=[
    #         model.init(jax.random.PRNGKey(0), jnp.ones(7))["params"] for _ in range(2)
    #     ],
    #     optimizer=optax.sgd(0.01),
    # )
    # self_play(
    #     train_state=train_state,
    #     n_hands=1_000_000,
    #     n_samples=10,
    #     key=jax.random.PRNGKey(1234),
    # )
